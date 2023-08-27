# https://www.openslr.org/33

from data_reader.base_reader import BaseReader
import os
from tqdm import tqdm
import pickle
from utils.whisper_duration_auto_tag import WhisperDurationTagger
from pypinyin import pinyin, lazy_pinyin, Style

class Openslr68(BaseReader):
    def __init__(self, train=True,  key_filter=None) -> None:
        super().__init__(train, key_filter)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp1'
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.MODEL_SIZE = str(self.config['model_size'])
        self.DISABLE_TQDM = bool(self.config['DISABLE_TQDM'])
        self.audio_transcript_pair_list = self.get_dataset(train)

    def get_dataset(self, train):
        dataset_txt = 'train' if train else 'test'
        self.text_path = self.config['path']+dataset_txt+'/TRANS.txt'
        self.audio_path = self.config['path']+dataset_txt+'/'
        self.audio_path_map = self.parse_path_map_txt(self.config['path']+dataset_txt+'.scp')
        text_data = self.parse_txt(self.text_path)
        return self.get_audio_file_list(text_data, self.audio_path, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'{dataset_txt}_audio_openslr68')

    def get_audio_file_list(self, text_data, audio_path, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000, save_name = None):
        save_dir = self.pickle_path+'/'+save_name
        if save_name is not None:
            if os.path.isfile(save_dir):
                with open(save_dir,'rb') as out_data:
                    return pickle.load(out_data)
        print('Because this is the first time you read this dataset, please wait for data index')
        audio_transcript_pair_list = []
        wav_files_not_exist = []
        text_or_audio_exceed_size = []
        
        for data in tqdm(text_data, disable=self.DISABLE_TQDM):
            utteranceID, text = data
            if utteranceID not in self.audio_path_map:
                print(f'[Warn] {utteranceID} not found in scp file, ignorn it')
                continue
            wav_file = self.audio_path_map[utteranceID]
            wav_file = wav_file.replace('wav/train/', '').replace('wav/test/', '')
            if not os.path.isfile(audio_path+wav_file):
                print(f'[Warn] file {audio_path+wav_file} not exist, ignorn it')
                wav_files_not_exist.append(wav_file)
                continue
            duration_start, duration_end = [], []
            # audio = self.load_wave(audio_path+wav_file, sample_rate=sample_rate)[0]
            # if len(text) > text_max_length or len(audio) > audio_max_sample_length:
            #     print(f'{text}, {wav_file}, {len(text)} > {text_max_length} or {len(audio)} > {audio_max_sample_length}')
            #     text_or_audio_exceed_size.append(id)
            #     continue
            audio_file_size = os.path.getsize(audio_path+wav_file)
            if len(text) > text_max_length or audio_file_size > audio_max_sample_length*2:
                text_or_audio_exceed_size.append(id)
                print(f'{text}, {wav_file}, {len(text)} > {text_max_length} or {audio_file_size} > {audio_max_sample_length*2}')
                continue
            audio_transcript_pair_list.append((str(wav_file), text, duration_start, duration_end))
        # print all id that ignored
        print(f'There are {len(wav_files_not_exist)} wav files not exist, they are {wav_files_not_exist}')
        print(f'There are {len(text_or_audio_exceed_size)} text or audio exceed size, they are {text_or_audio_exceed_size}')
        # cache to pickle 
        if save_name is not None:
            if not os.path.exists(self.pickle_path):
                os.makedirs(self.pickle_path)
            with open(save_dir,'wb') as in_data:
                pickle.dump(audio_transcript_pair_list,in_data,pickle.HIGHEST_PROTOCOL)
        return audio_transcript_pair_list
        
    def parse_txt(self, file_name):
        # example output: 38_5716_20170914202211.wav    嗨天气寒冷记得添衣保暖哦
        data = []
        start_flag = True
        with open(file_name, 'r') as f:
            for line in f:
                UtteranceID, SpeakerID, Transcription = line.split('\t')
                if start_flag:
                    assert UtteranceID == 'UtteranceID' and SpeakerID == 'SpeakerID' and Transcription == 'Transcription\n', f'{UtteranceID}, {SpeakerID}, {Transcription}'
                    start_flag = False
                else:
                    try:
                        Transcription = self.filter_chinese(Transcription)
                        data.append((UtteranceID, Transcription))
                    except:
                        print(f'for text={Transcription} ignore it because no chinese detected')
        return data
    
    def parse_path_map_txt(self, file_name):
        # example output: 38_5716_20170914202211.wav    嗨天气寒冷记得添衣保暖哦
        data = {}
        start_flag = True
        with open(file_name, 'r') as f:
            for line in f:
                UtteranceID, path = line.split('\t')
                if '\n' in path:
                    path = path.replace('\n', '')
                data[UtteranceID] = path
        return data
    
    def get_naive_item(self, idx):
        pair = super().__getitem__(idx)
        wav_path, text, dur_start, dur_end = pair
        return wav_path, text, dur_start, dur_end
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        wav_path, text, dur_start, dur_end = pair
        # hanzi_words = [i if i != ' ' else 'SP' for i in text]
        # pinyin = [lazy_pinyin(i)[0] if i != ' ' else 'SP' for i in text]
        # tone = [i[-1] if len(i)>0 and i[-1] in ['1','2','3','4','5'] else '0' for i in [lazy_pinyin(i, style=Style.FINALS_TONE3)[0] for i in text]]
        hanzi_words = [i if i != ' ' else 'SP' for i in text.replace(' ', '')]
        pinyin = [lazy_pinyin(i)[0] if i != ' ' else 'SP' for i in text.replace(' ', '')]
        tone = [i[-1] if len(i)>0 and i[-1] in ['1','2','3','4','5'] else '0' for i in [lazy_pinyin(i, style=Style.FINALS_TONE3)[0] for i in text.replace(' ', '')]]
        dur_start = [f'{i:.2f}' for i in dur_start]
        dur_end = [f'{i:.2f}' for i in dur_end]
        # assert len(hanzi_words) == len(pinyin) and len(tone) == len(dur_start) and \
        #     len(dur_end) == len(dur_start) and len(hanzi_words) == len(tone)
        assert len(hanzi_words) == len(pinyin) and len(tone) == len(pinyin) and len(hanzi_words) == len(tone)
        output = {
            'audio': self.audio_path+wav_path,
            'hanzi': hanzi_words,
            'pinyin': pinyin,
            'tone': tone,
            # 'start': dur_start,
            # 'end': dur_end,
        }
        if self.key_filter == None:
            return output
        return {key: value for key, value in output.items() if key in self.key_filter}

if __name__ == '__main__':
    def print_data(hanzi_words, pinyin, tone, dur_start, dur_end):
        assert len(hanzi_words) == len(pinyin) and len(tone) == len(dur_start) and \
            len(dur_end) == len(dur_start) and len(hanzi_words) == len(tone)
        for i in range(len(hanzi_words)):
            print(f'{hanzi_words[i]}:{pinyin[i]:<6}[{tone[i]}] ({dur_start[i]}->{dur_end[i]})')
    
    openslr_68_test = Openslr68(train=False)
    openslr_68_train = Openslr68()
    print(len(openslr_68_train), len(openslr_68_test))
    data = openslr_68_train[56]
    print(data)
    # print_data(data['hanzi'], data['pinyin'], data['tone'], data['start'], data['end'])
   