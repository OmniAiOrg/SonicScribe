# https://www.openslr.org/33

import json
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

class Openslr47(BaseReader):
    def __init__(self, train=True) -> None:
        super().__init__(train)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp-no-duration'
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.MODEL_SIZE = str(self.config['model_size'])
        self.DISABLE_TQDM = bool(self.config['DISABLE_TQDM'])
        self.audio_transcript_pair_list = self.get_dataset(train)

    def get_dataset(self, train):
        dataset_txt = 'train' if train else 'test'
        self.text_path = self.config['path']+'set1_transcript.json'
        self.audio_path = self.config['path']+'audio_files/'
        text_data_list = self.parse_txt(self.text_path)
        text_data_train, text_data_test = text_data_list[:50000], text_data_list[50000:]
        text_data = text_data_train if train else text_data_test
        return self.get_audio_file_list(text_data, self.audio_path, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'{dataset_txt}_audio_openslr47')
        
    def audio_path_to_wav_list(self, dir_path):
        sub_dirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        wav_files = []
        for sub_dir in sub_dirs:
            wav_files.extend(glob.glob(os.path.join(sub_dir, '*.wav')))
        wav_files = [os.path.relpath(f, dir_path) for f in wav_files]
        return wav_files

    def get_audio_file_list(self, text_data, audio_path, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000, save_name = None):
        save_dir = self.pickle_path+'/'+save_name
        if save_name is not None:
            if os.path.isfile(save_dir):
                with open(save_dir,'rb') as out_data:
                    return pickle.load(out_data)
        print('Because this is the first time you read this dataset, please wait for data index')
        audio_transcript_pair_list = []
        # duration_tagger = WhisperDurationTagger(self.MODEL_SIZE, True)
        wav_files_not_exist = []
        text_or_audio_exceed_size = []
        for data in tqdm(text_data, disable=self.DISABLE_TQDM):
            id, file, user_id, text, length = [data[s] for s in ['id', 'file', 'user_id', 'text', 'length']]
            wav_file = f'{file[0]}/{file[0:2]}/{file}'
            if not os.path.isfile(audio_path+wav_file):
                wav_files_not_exist.append(wav_file)
                continue
            # duration = duration_tagger.predict(text.replace(' ', ''), audio_path+wav_file)
            # if len(duration) == 0:
            #     continue
            # duration_start = [st.start for st in duration]
            # duration_end = [st.end for st in duration]
            duration_start, duration_end = [], []
            audio = self.load_wave(audio_path+wav_file, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print(f'{text}, {wav_file}, {len(text)} > {text_max_length} or {len(audio)} > {audio_max_sample_length}')
                text_or_audio_exceed_size.append(id)
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
        with open(file_name, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
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
        return {
            'audio': self.audio_path+wav_path,
            'hanzi': hanzi_words,
            'pinyin': pinyin,
            'tone': tone,
            # 'start': dur_start,
            # 'end': dur_end,
        }

if __name__ == '__main__':
    def print_data(hanzi_words, pinyin, tone, dur_start, dur_end):
        assert len(hanzi_words) == len(pinyin) and len(tone) == len(dur_start) and \
            len(dur_end) == len(dur_start) and len(hanzi_words) == len(tone)
        for i in range(len(hanzi_words)):
            print(f'{hanzi_words[i]}:{pinyin[i]:<6}[{tone[i]}] ({dur_start[i]}->{dur_end[i]})')
    
    openslr_33_test = Openslr47(train=False)
    openslr_33_train = Openslr47()
    print(len(openslr_33_train), len(openslr_33_test))
    data = openslr_33_train[56]
    print(data)
    # print_data(data['hanzi'], data['pinyin'], data['tone'], data['start'], data['end'])
   