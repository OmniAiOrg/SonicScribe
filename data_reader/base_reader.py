# base reader
import torch
import torchaudio
import torchaudio.transforms as at
from utils.chinese_to_pinyin import is_chinese
from utils.load_checkpoint import get_config

class BaseReader(torch.utils.data.Dataset):
    def __init__(self, train, key_filter=None) -> None:
        self.dataset_name = type(self).__name__
        self.config = self.get_config()
        if train:
            self.print_config()
        self.train = train
        self.audio_transcript_pair_list = None
        self.path:str = self.config['path']
        self.dummy_reader = False # if True, get_item will not return wav
        self.key_filter = key_filter
        
    def set_dummy(self, dummy: bool):
        self.dummy_reader = dummy
        
    def load_wave(self, wave_path, sample_rate:int=16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform
        
    def get_config(self):
        all_config = get_config('data_reader/dataset_config.yaml')
        base_config = all_config['BaseReader']
        specific_config = all_config[self.dataset_name]
        for k in specific_config.keys():
            base_config[k] = specific_config[k]
        return base_config
    
    def print_config(self):
        print('=== Dataset Info ===')
        keys = ['name', 'summary', 'category', 'license', 'website']
        for k in keys:
            print(f'{k.upper():<10}: {self.config[k]}')
            
    def filter_chinese(self, input:str) -> str:
        output=''
        for i in input:
            if is_chinese(i):
                output += i
        assert len(output) > 0, input
        return output
        
    
    def __len__(self):
        return len(self.audio_transcript_pair_list)
    
    def __getitem__(self, idx):
        return self.audio_transcript_pair_list[idx]