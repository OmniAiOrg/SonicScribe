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

class WhisperDurationTagger:
    def __init__(self, model_name='tiny', force_cpu=False, download_root='./assets/whisper_checkpoint/') -> None:
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.model = whisper.load_model(model_name, download_root=download_root).to(device)
        self.language = "zh"
        self.tokenizer = get_tokenizer(self.model.is_multilingual)
    
    def predict(self, text: str, wav_path: str) -> list[float]:
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
            initial_prompt=text
        )
        for segment in result["segments"]:
            for timing in segment["words"]:
                # assert timing["start"] < timing["end"]
                print(f'{timing["word"]}: {timing["start"]} -> {timing["end"]}')

        return result
    
if __name__ == '__main__':
    tagger = WhisperDurationTagger(download_root='/mnt/private_cq/whisper_checkpoint')
    tagger.predict('该 土 地 拍 卖 价 刷 新 了 杨 浦 区 土 地 最 贵 单 价 记 录', './assets/sample_audio/33-Aishell/data_aishell/wav/dev/S0726/BAC009S0726W0171.wav')
    # test_transcribe(
    #     'tiny', 
    #     './assets/sample_audio/33-Aishell/data_aishell/wav/dev/S0726/BAC009S0726W0171.wav',
    #     '该土地拍卖价刷新了杨浦区土地最新贵单价记录'
    #     )