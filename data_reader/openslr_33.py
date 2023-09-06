# https://www.openslr.org/33

from data_reader.base_reader import BaseReader
import torch
from pathlib import Path
import os
import numpy as np
from torch import nn
import whisper
import torchaudio
import torchaudio.transforms as at
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple
import glob
# from utils.general_dataloader import SonicData
from utils.pre_tokenizers import *
from utils.whisper_duration_auto_tag import WhisperDurationTagger
from pypinyin import pinyin, lazy_pinyin, Style

class Openslr33(BaseReader):
    def __init__(self, train=True, key_filter=None) -> None:
        super().__init__(train, key_filter)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp4'
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.MODEL_SIZE = str(self.config['model_size'])
        self.DISABLE_TQDM = bool(self.config['DISABLE_TQDM'])
        self.audio_transcript_pair_list = self.get_dataset(train)
        
        # self.initials_tokenizer = initials_tokenizer
        # self.finals_tokenizer = finals_tokenizer
        # self.word_tokenizer = word_tokenizer

    def get_dataset(self, train):
        dataset_txt = 'train' if train else 'test'
        self.text_path = self.config['path']+'transcript/aishell_transcript_v0.8.txt'
        self.audio_path = self.config['path']+f'wav/{dataset_txt}/'
        text_data = self.parse_txt(self.text_path)
        return self.get_audio_file_list(text_data, self.audio_path, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'{dataset_txt}_audio_openslr33')
        
    def audio_path_to_wav_list(self, dir_path):
        sub_dirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        wav_files = []
        for sub_dir in sub_dirs:
            wav_files.extend(glob.glob(os.path.join(sub_dir, '*.wav')))
        wav_files = [os.path.relpath(f, dir_path) for f in wav_files]
        return wav_files
        
    def wav_path_to_id(self, wav_path):
        file_name = os.path.basename(wav_path)
        id = os.path.splitext(file_name)[0]
        return id

    def get_audio_file_list(self, text_data, audio_path, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000, save_name = None):
        save_dir = self.pickle_path+'/'+save_name
        if save_name is not None:
            if os.path.isfile(save_dir):
                with open(save_dir,'rb') as out_data:
                    return pickle.load(out_data)
        print('Because this is the first time you read this dataset, please wait for data index')
        audio_transcript_pair_list = []
        wav_files = self.audio_path_to_wav_list(audio_path)
        print(f'There are {len(wav_files)} wav files in {audio_path}')
        # print(wav_files[:10])
        # exit(0)
        duration_tagger = WhisperDurationTagger(self.MODEL_SIZE)
        ids_not_in_text = []
        text_or_audio_exceed_size = []
        for wav_file in tqdm(wav_files, disable=self.DISABLE_TQDM):
            id = self.wav_path_to_id(wav_file)
            if id not in text_data:
                print(f'{id} not in {self.text_path}')
                ids_not_in_text.append(id)
                continue
            text = text_data[id]
            if '\n' in text:
                text = text.split('\n')[0]
            text = ''.join(text.split(' '))
            duration = duration_tagger.predict(text, audio_path+wav_file)
            if len(duration) == 0:
                continue
            duration_start = [st.start for st in duration]
            duration_end = [st.end for st in duration]
            audio = self.load_wave(audio_path+wav_file, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print(f'{text}, {wav_file}, {len(text)} > {text_max_length} or {len(audio)} > {audio_max_sample_length}')
                text_or_audio_exceed_size.append(id)
                continue
            audio_transcript_pair_list.append((str(wav_file), text, duration_start, duration_end))
        # print all id that ignored
        print(f'There are {len(ids_not_in_text)} ids not in text, they are {ids_not_in_text}')
        print(f'There are {len(text_or_audio_exceed_size)} text or audio exceed size, they are {text_or_audio_exceed_size}')
        # cache to pickle 
        if save_name is not None:
            if not os.path.exists(self.pickle_path):
                os.makedirs(self.pickle_path)
            with open(save_dir,'wb') as in_data:
                pickle.dump(audio_transcript_pair_list,in_data,pickle.HIGHEST_PROTOCOL)
        return audio_transcript_pair_list
        
    def parse_txt(self, file_name):
        data = {}
        with open(file_name, 'r') as f:
            for line in f:
                id, content = line.split(' ', 1)
                data[id] = content
        return data
    
    def get_naive_item(self, idx):
        pair = super().__getitem__(idx)
        wav_path, text, dur_start, dur_end = pair
        return wav_path, text, dur_start, dur_end
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        wav_path, text, dur_start, dur_end = pair
        hanzi_words = [i for i in text]
        pinyin = [lazy_pinyin(i)[0] for i in text]
        tone = [i[-1] if len(i)>0 and i[-1] in ['1','2','3','4','5'] else '0' for i in [lazy_pinyin(i, style=Style.FINALS_TONE3)[0] for i in text]]
        dur_start = [f'{i:.2f}' for i in dur_start]
        dur_end = [f'{i:.2f}' for i in dur_end]
        assert len(hanzi_words) == len(pinyin) and len(tone) == len(dur_start) and \
            len(dur_end) == len(dur_start) and len(hanzi_words) == len(tone)
        output = {
            'audio': self.audio_path+wav_path,
            'hanzi': hanzi_words,
            'pinyin': pinyin,
            'tone': tone,
            'start': dur_start,
            'end': dur_end
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
    
    openslr_33_test = Openslr33(train=False)
    openslr_33_train = Openslr33()
    print(len(openslr_33_train), len(openslr_33_test))
    data = openslr_33_train[56]
    print(data['audio'])
    import shutil
    shutil.copyfile(data['audio'], 'logs/test.wav')
    print_data(data['hanzi'], data['pinyin'], data['tone'], data['start'], data['end'])
    '''
    环:huan  [2] (0.32->0.80)
    比:bi    [3] (0.80->1.02)
    六:liu   [4] (1.02->1.16)
    月:yue   [4] (1.16->1.38)
    份:fen   [4] (1.38->1.66)
    上:shang [4] (1.66->1.90)
    升:sheng [1] (1.90->2.24)
    百:bai   [3] (2.24->2.30)
    分:fen   [1] (2.30->2.56)
    之:zhi   [1] (2.56->2.90)
    一:yi    [1] (2.90->3.20)
    十:shi   [2] (3.20->3.44)
    五:wu    [3] (3.44->3.66)
    点:dian  [3] (3.66->3.86)
    六:liu   [4] (3.86->4.14)
    '''
    