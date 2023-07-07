# base reader
import torch
import yaml
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
from whisper.tokenizer import get_tokenizer, Tokenizer

class BaseReader(torch.utils.data.Dataset):
    def __init__(self, train) -> None:
        self.dataset_name = type(self).__name__
        self.config = self.get_config()
        self.print_config()
        self.train = train
        self.audio_transcript_pair_list = None
        self.path:str = self.config['path']
        
    def load_wave(self, wave_path, sample_rate:int=16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform
        
    def get_config(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'dataset_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config[self.dataset_name]
    
    def print_config(self):
        print('=== Dataset Info ===')
        keys = ['name', 'summary', 'category', 'license', 'website']
        max_key_len = max([len(key) for key in keys])
        for k in keys:
            print(f'{k.upper():<10}: {self.config[k]}')
        
    
    def __len__(self):
        return len(self.audio_transcript_pair_list)
    
    def __getitem__(self, idx):
        return self.audio_transcript_pair_list[idx]