# https://www.openslr.org/33

from data_reader.base_reader import BaseReader
import torch
from pathlib import Path
import os
import numpy as np
from torch import nn
import torchaudio
import torchaudio.transforms as at
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple
import glob

class Openslr33(BaseReader):
    def __init__(self, train=True) -> None:
        super().__init__(train)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp'
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.audio_transcript_pair_list = self.get_dataset(train)
        
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
        # print(wav_files[:10])
        # exit(0)
        for wav_file in tqdm(wav_files):
            id = self.wav_path_to_id(wav_file)
            assert id in text_data, f'{id} not in {self.text_path}'
            text = text_data[id]
            text = ''.join(text.split(' '))
            audio = self.load_wave(audio_path+wav_file, sample_rate=sample_rate)[0]
            assert len(text) <= text_max_length and len(audio) <= audio_max_sample_length, f'{text}, {wav_file}, {len(text)}, {len(audio)}'
            audio_transcript_pair_list.append((str(wav_file), text))
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
                
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        wav_path, text = pair
        wav_path = self.audio_path + wav_path
        if '\n' in text:
            text = text.split('\n')[0]
        return wav_path, text
        
    
    

if __name__ == '__main__':
    openslr_33_train = Openslr33()
    openslr_33_test = Openslr33(train=False)
    print(openslr_33_test[0])
    