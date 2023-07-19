'''
Whisper had improved its ability in token level duration prediction.
This prediction may predict multiple chinese characters into single token,
but most cases it will predict single Chinese chiaracter into multiple tokens,
which is acceptable.
So here we use Whisper to auto tag wav audio file into character and duration.
We may input the actual label from the dataset as prompt, in which case the 
predicted characters would be the same.
And there may be a small trick, which is add spaces between every two Chinese
characters, so the prediction will prefer to predict single chinese characters.
The we could calculate the character level duration, which may be not accurate,
so we also need to distinguish whether the predicted duration is acceptable.
finally, we will return the predicted list of duration, which may contain
-100, not predicted.
'''

import torch
import whisper
from whisper.tokenizer import get_tokenizer
from dataclasses import dataclass
from utils.simplified_chinese_tokenizer import traditional_to_simplified

class WhisperDurationTagger:
    def __init__(self, model_name='tiny', force_cpu=False, download_root='./assets/whisper_checkpoint/') -> None:
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.model = whisper.load_model(model_name, download_root=download_root).to(device)
        self.language = "zh"
        self.tokenizer = get_tokenizer(self.model.is_multilingual)
        self.get_chinese_exchange_pair()
        
    def get_chinese_exchange_pair(self):
        self.origin_to_dummy_pair={}
        self.dummy_to_origin_pair={}
        with open('./assets/chinese_words_with_single_token.txt', 'r') as f:
            for line in f:
                if line.endswith('\n'):
                    line = line[:-1]
                _, id, org, ex = line.split(' ')
                self.origin_to_dummy_pair[org] = ex
                self.dummy_to_origin_pair[ex] = org
    @dataclass
    class TransChar:
        word: str
        start: float
        end: float
        
    def naive_predict(self, text: str, wav_path: str) -> list[TransChar]:
        '''
        The entrance of this function.
        Input: 
            text: a string of Chinese text
            wav_path: path of input audio
        Output:
            prediction: a list of float value, map to each Chinese chracters one by one
        '''
        result = self.model.transcribe(
            wav_path, 
            language=self.language, 
            temperature=0.0, 
            word_timestamps=True,
            initial_prompt=text,
            condition_on_previous_text = False,
        )
        output = []
        for segment in result["segments"]:
            for timing in segment["words"]:
                assert timing["start"] < timing["end"]
                output.append(self.TransChar(
                    traditional_to_simplified(timing["word"]),
                    float(timing["start"]),
                    float(timing["end"])))
        return output
        
    def pre_predict(self, text: str, wav_path: str) -> list[TransChar]:
        '''
        Iteration for word with multiple Chinese characters, add a blank between the word;
        Break until no word contains multiple Chinese characters.
        Then check whether there are words not continuous, 
        Then check whether there is a character not in prompt, if so, remove it.
        
        '''
        output = self.naive_predict(text, wav_path)
        # if two or more chinese characters have a single token, change the word
        new_text = text
        replace_flag = False
        for tc in output:
            if tc.word in self.origin_to_dummy_pair:
                new_word = self.origin_to_dummy_pair[tc.word]
                new_text = new_text.replace(tc.word, new_word)
                replace_flag = True
        if not replace_flag:
            return output
        # after replace with the word cannot be represented with single token, try again
        output = self.naive_predict(new_text, wav_path)
        replace_flag = False
        for tc in output:
            if tc.word in self.origin_to_dummy_pair:
                tc.word = self.origin_to_dummy_pair[tc.word]
                replace_flag = True
        if not replace_flag:
            return output
        # if still has multiple char with single token, rebuild from predicted
        new_text = ''.join([tc.word for tc in output])
        output = self.naive_predict(new_text, wav_path)
        return output
    
    def match(self, output: list[TransChar], text: str) -> list[TransChar]:
        
        return output
    
    def predict(self, text: str, wav_path: str) -> list[TransChar]:
        '''
        Match the pre predicted time list with the text you used as prompt,
        if they can not be matched, return empty list.
        The text match use pinyin to compare
        '''
        output = self.pre_predict(text, wav_path)
        output = self.match(output, text)
        return output
        
    
if __name__ == '__main__':
    tagger = WhisperDurationTagger('tiny', download_root='/mnt/private_cq/whisper_checkpoint')
    output = tagger.predict('如果云层是天空的一封信', './assets/sample_audio/opencpop/2003000102.wav')
    # output = tagger.predict('该土地拍卖价刷新了杨浦区 土地最贵单价记录', './assets/sample_audio/33-Aishell/data_aishell/wav/dev/S0726/BAC009S0726W0171.wav')
    for sc in output:
        print(f'{sc.word}: {sc.start} -> {sc.end}')
    '''
    如: 0.0 -> 0.58
    果: 0.58 -> 0.88
    云: 0.88 -> 1.2
    层: 1.2 -> 1.56
    是: 1.56 -> 1.82
    天: 1.82 -> 2.18
    空: 2.18 -> 2.48
    的: 2.48 -> 2.84
    一: 2.84 -> 3.1
    封: 3.1 -> 3.36
    信: 3.36 -> 4.1
    '''