'''
All data reader will not convert original symbol to tokens. Instead, all tokenizer will
work in this dataloader. This means mask could also be generatted in here.
'''

import dataclasses
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple

import torchaudio
import torchaudio.transforms as at
from model.chinese_token_embeddings import ChineseTokenEmbedding
from model.pinyin_token_embeddings import PinyinTokenEmbedding
from utils.tokenizers import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, BatchSampler
import whisper
from utils.tokenizers import word_tokenizer, pinyin_tokenizer, note_tokenizer, tone_tokenizer, slur_tokenizer, duration_tokenizer

dataset_keys = [
    'audio', # path to audio
    'hanzi', # Chinese characterd with type list[str]
    'pinyin', # list[str]
    'note', # note like A#3/Bb3
    'tone', # 1,2,3,4,5 for 'ˉ', 'ˊ', 'ˇ', 'ˋ', ' '. 
    'slur', # finals have sustained sound with diffrent note
    'start', # start time in sec for this hanzi
    'end' # end time in sec for this hanzi
    ]
    
@dataclass
class SonicBatch:
    mel: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    hanzi: Optional[Tensor] = None
    hanzi_label: Optional[Tensor] = None
    note: Optional[Tensor] = None
    note_label: Optional[Tensor] = None
    start: Optional[Tensor] = None
    start_label: Optional[Tensor] = None
    end: Optional[Tensor] = None
    end_label: Optional[Tensor] = None
    slur: Optional[Tensor] = None
    slur_label: Optional[Tensor] = None
    pinyin:Optional[Tensor] = None
    pinyin_label: Optional[Tensor] = None
    tone:Optional[Tensor] = None
    tone_label: Optional[Tensor] = None
    
def sonic_batch_to_shape(sonic_batch: SonicBatch):
    field_names = [field.name for field in dataclasses.fields(SonicBatch)]
    sonic_batch_shape = {}
    for field_name in field_names:
        value = getattr(sonic_batch, field_name)
        if value is not None:
            sonic_batch_shape[field_name] = value.shape
    return sonic_batch_shape
    

'''
Process list of SonicData to SonicBatch with padding.
In this call function, the input may be from multiple different dataset,
so a list may contain both None and not None values. 
'''
class WhisperDataCollatorWithPadding:
    def __init__(self, label_pad=-1, pad=-2, unkown=-3, n_state=384, model='tiny', device='cpu') -> None:
        self.const_pad_label = label_pad
        self.const_pad = pad
        self.unknow = unkown
        self.word_tokenizer = word_tokenizer
        self.pinyin_tokenizer = pinyin_tokenizer
        self.note_tokenizer = note_tokenizer
        self.tone_tokenizer = tone_tokenizer
        self.slur_tokenizer = slur_tokenizer
        self.duratoin_tokenizer = duration_tokenizer
        all_config = get_config('data_reader/dataset_config.yaml')
        base_config = all_config['BaseReader']
        self.SAMPLE_RATE = base_config['SAMPLE_RATE']
        # there are two special embedding should be initialized here
        self.word_embedding = ChineseTokenEmbedding(5000, self.word_tokenizer, model)
        self.pinyin_embedding = PinyinTokenEmbedding(self.pinyin_tokenizer, model)
        self.device = device
        
    def load_wave(self, wave_path, sample_rate:int=16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform
        
    def audio_to_mel(self, audio):
        audio = self.load_wave(audio, sample_rate=self.SAMPLE_RATE)
        audio = audio.flatten()
        assert audio.shape[-1] < whisper.audio.N_SAMPLES # or it will be cut
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel
        
    def get_keys_mask(self, data: dict) -> list[bool]:
        return [dataset_keys[i] in data.keys() for i in range(len(dataset_keys))]
    
    def feature_encode(self, data:dict, size_of_input:int, mask:bool, task:str, features: Dict[str, list], aim_task:str, tokenizer:NaiveTokenizer) -> None:
        if task == aim_task:
            if task not in features:
                features[task] = []
                features[task+'_label'] = []
            if mask:
                words = [*tokenizer.sot_task_so_on] + tokenizer.encode(data[task], default=self.unknow)
            else:
                words = (len(tokenizer.sot_task_so_on) + size_of_input)*[0]
            features[task].append(words)
            features[task+'_label'].append(words[1:]+[tokenizer.eot])
        
    def __call__(self, input: list[dict]) -> SonicBatch:
        features: Dict[str, list] = {'mel':[], 'mask':[]}
        feature_lengths: list[int] = []
        for data in input:
            assert len(set(data.keys()) - set(dataset_keys)) == 0, 'not all keys in data legal'
            assert 'pinyin' in data.keys() or 'note' in data.keys()
            # 0. update Chinese characters table, and also transcribe traditional to simplified
            self.word_embedding.auto_update(data['hanzi'])
            # 1. get size of dataset
            size_of_input = len(data['pinyin'] or data['note'])
            feature_lengths.append(size_of_input+len(self.pinyin_tokenizer.sot_task_so_on))
            # 2. get keys mask
            mask = self.get_keys_mask(data)
            features['mask'].append(mask)
            # 3. go through mask, encode all in mask
            for i in range(len(mask)):
                task = dataset_keys[i]
                if task == 'audio':
                    mel = self.audio_to_mel(data['audio'])
                    features['mel'].append(mel)
                    continue
                self.feature_encode(data, size_of_input, mask[i], task, features, 'hanzi', self.word_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'pinyin', self.pinyin_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'note', self.note_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'tone', self.tone_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'slur', self.slur_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'start', self.duratoin_tokenizer)
                self.feature_encode(data, size_of_input, mask[i], task, features, 'end', self.duratoin_tokenizer)

        features['mel'] = torch.concat([m[None, :] for m in features['mel']])
        features['mask'] = torch.Tensor(features['mask'])
        max_feature_len = max(feature_lengths)

        for k, v in features.items():
            if k in ['mel', 'mask']:
                continue
            if 'label' in k:
                constant_values = self.const_pad_label
            else:
                constant_values = self.const_pad
            feature_lengths_ = [len(p) if p is not None else 0 for p in v]
            # print(k, v)
            assert feature_lengths_ == feature_lengths, f'{k}, current {feature_lengths_} not equals to {feature_lengths}'
            padded_data = [torch.nn.functional.pad(torch.tensor(f), (0, max_feature_len - f_len), value=constant_values) 
               for f, f_len in zip(v, feature_lengths)]
            features[k] = torch.stack(padded_data, dim=0)

        sonic_batch = SonicBatch()
        [setattr(sonic_batch, field_name, v) for field_name, v in features.items()]

        return sonic_batch
    
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_with_weights):
        self.datasets = [ds for ds, _ in datasets_with_weights]
        self.weights = [weight for _, weight in datasets_with_weights]
        self.lengths = [len(ds) for ds in self.datasets]
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        self.cum_weights = torch.tensor(self.weights).cumsum(dim=0).tolist()

    def __getitem__(self, index):
        dataset_idx = self._get_random_dataset_index()
        selected_dataset = self.datasets[dataset_idx]
        return selected_dataset[index % self.lengths[dataset_idx]]

    def _get_random_dataset_index(self):
        rand_val = torch.rand(1).item()
        for i, cum_weight in enumerate(self.cum_weights):
            if rand_val < cum_weight:
                return i
        return len(self.cum_weights) - 1

    def __len__(self):
        return max(self.lengths)
    
    
if __name__ == '__main__':
    from data_reader.opencpop import OpenCpop
    from data_reader.openslr_33 import Openslr33
    dataset_a = OpenCpop(train=False)
    dataset_b = Openslr33(train=False)
    print(len(dataset_a), len(dataset_b), flush=True)
    weighted_dataset = WeightedDataset([(dataset_a, 3), (dataset_b, 3)])
    
    loader = DataLoader(weighted_dataset, 
                            batch_size=10,
                            shuffle=True,
                            collate_fn=WhisperDataCollatorWithPadding()
                          )
    for b in loader:
        print(b)
        print(sonic_batch_to_shape(b))
        break