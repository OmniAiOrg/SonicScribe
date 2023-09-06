# https://www.openslr.org/33

import random
from data_reader.base_reader import BaseReader
import os
from tqdm import tqdm
import pickle
# from utils.whisper_duration_auto_tag import WhisperDurationTagger
from pypinyin import pinyin, lazy_pinyin, Style

class Openslr38(BaseReader):
    def __init__(self, train=True,  key_filter=None) -> None:
        super().__init__(train, key_filter)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp1'
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.MODEL_SIZE = str(self.config['model_size'])
        self.DISABLE_TQDM = bool(self.config['DISABLE_TQDM'])
        self.audio_path = self.config['path']
        self.audio_transcript_pair_list = self.get_dataset(train)
        

    def get_dataset(self, train):
        audio_transcript_pair_list = self.get_audio_file_list(self.audio_path, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'audio_openslr38')
        def split_train_test(data, test_ratio, random_seed):
            random.seed(random_seed)
            random.shuffle(data)
            split_index = int(len(data) * test_ratio)
            train_data = data[split_index:]
            test_data = data[:split_index]
            return train_data, test_data
        train_data, test_data = split_train_test(audio_transcript_pair_list, 0.01, 0)
        return train_data if train else test_data

    def get_audio_file_list(self, path, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000, save_name = None):
        save_dir = self.pickle_path+'/'+save_name
        if save_name is not None:
            if os.path.isfile(save_dir):
                with open(save_dir,'rb') as out_data:
                    return pickle.load(out_data)
        print('Because this is the first time you read this dataset, please wait for data index')
        audio_transcript_pair_list = []
        wav_files_not_exist = []
        text_or_audio_exceed_size = []
        
        def get_file_names_without_extension(path):
            file_names = []
            for file in os.listdir(path):
                if file.endswith(".wav"):
                    file_names.append(os.path.splitext(file)[0])
            return file_names
        
        def read_single_line_from_file(txt_file_path):
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()
                assert len(lines) == 1, f"more than one line found in {txt_file_path}"
                return lines[0].strip()

        file_names = get_file_names_without_extension(path)
        audio_path = path
        for file_name in tqdm(file_names, disable=self.DISABLE_TQDM):
            wav_file = file_name+".wav"
            txt_file = file_name+".txt"
            if not os.path.isfile(audio_path+wav_file) or not os.path.isfile(audio_path+txt_file):
                print(f'[Warn] file {audio_path+wav_file} or {audio_path+txt_file} not exist, ignorn it')
                wav_files_not_exist.append(file_name)
                continue
            duration_start, duration_end = [], []
            audio_file_size = os.path.getsize(audio_path+wav_file)
            text = read_single_line_from_file(audio_path+txt_file)
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
    
    openslr_test = Openslr38(train=False)
    openslr_train = Openslr38()
    print(len(openslr_train), len(openslr_test))
    data = openslr_train[0]
    print(data)
    import shutil
    shutil.copyfile(data['audio'], 'logs/test.wav')
    # print_data(data['hanzi'], data['pinyin'], data['tone'], data['start'], data['end'])
   