# base reader
import torch
import torchaudio
import torchaudio.transforms as at
from utils.chinese_to_pinyin import is_chinese
from utils.load_checkpoint import get_config
from torchaudio.functional import vad
from torchaudio.transforms import Vol
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
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE']) if 'SAMPLE_RATE' in self.config else 16000
        self.vol_transform = Vol(gain=0.6)  
        
    def set_dummy(self, dummy: bool):
        self.dummy_reader = dummy
        
    def load_wave(self, wave_path, sample_rate:int=16000, val_norm=False) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        if val_norm:
            waveform = self.vol_transform(waveform)
        return waveform
    
    def sox_vad(self, wave_form):
        effects = [
            ['silence', '1', '0.1', '1%'], # remove silence from head until more than 0.1s of audio with more than 3% volume
            ['reverse'],
            ['silence', '1', '0.1', '1%'],
            ['reverse'],
            ['norm']
        ]
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(wave_form, self.SAMPLE_RATE, effects, channels_first=True)
        assert sample_rate == self.SAMPLE_RATE
        return waveform
    
    def torchaudio_vad(self, wave_form):
        vad_waveform = vad(wave_form, sample_rate=self.SAMPLE_RATE, trigger_time=0.1, trigger_level=7, allowed_gap=0.1)
        reversed_waveform = torch.flip(vad_waveform, [1])
        reversed_vad_waveform = vad(reversed_waveform, sample_rate=self.SAMPLE_RATE, trigger_time=0.1, trigger_level=7, allowed_gap=0.1)
        waveform = torch.flip(reversed_vad_waveform, [1])
        return waveform
    
    def get_waveform(self, wave_path, trim=False, sox=True):
        waveform = self.load_wave(wave_path, sample_rate=self.SAMPLE_RATE)
        if trim and not sox:
            waveform = self.torchaudio_vad(waveform)
        if trim and sox:
            waveform = self.sox_vad(waveform)
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
    
    def __getitem__(self, idx) -> dict:
        return self.audio_transcript_pair_list[idx]