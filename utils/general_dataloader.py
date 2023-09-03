'''
All data reader will not convert original symbol to tokens. Instead, all tokenizer will
work in this dataloader. This means mask could also be generatted in here.
'''

from dataclasses import dataclass, fields
import math
import random
import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple

import torchaudio
import torchaudio.transforms as at
from data_reader.base_reader import BaseReader
from model.chinese_token_embeddings import ChineseTokenEmbedding
from model.pinyin_token_embeddings import PinyinTokenEmbedding
from utils.pre_tokenizers import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, BatchSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import whisper
from utils.pre_tokenizers import all_tokenizers, word_tokenizer, pinyin_tokenizer, note_tokenizer, tone_tokenizer, slur_tokenizer, duration_tokenizer
from utils.naive_tokenizer import get_tokenizer

dataset_keys = [
    'audio',  # path to audio
    'hanzi',  # Chinese characterd with type list[str]
    'pinyin',  # list[str]
    'note',  # note like A#3/Bb3
    'tone',  # 1,2,3,4,5 for 'ˉ', 'ˊ', 'ˇ', 'ˋ', ' '.
    'slur',  # finals have sustained sound with diffrent note
    'start',  # start time in sec for this hanzi
    'end'  # end time in sec for this hanzi
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
    pinyin: Optional[Tensor] = None
    pinyin_label: Optional[Tensor] = None
    tone: Optional[Tensor] = None
    tone_label: Optional[Tensor] = None


def get_field_names(cls: type, exclude: List[str] = []) -> List[str]:
    return [f.name for f in fields(cls) if f.name not in exclude]


def sonic_batch_to_shape(sonic_batch: SonicBatch):
    field_names = [field.name for field in fields(SonicBatch)]
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
    def __init__(self, model='tiny', num_workers=1) -> None:
        self.all_tokenizers = all_tokenizers

        self.word_tokenizer = word_tokenizer
        self.pinyin_tokenizer = pinyin_tokenizer
        self.note_tokenizer = note_tokenizer
        self.tone_tokenizer = tone_tokenizer
        self.slur_tokenizer = slur_tokenizer
        self.duratoin_tokenizer = duration_tokenizer

        self.const_pad_label = word_tokenizer.pad_label
        self.const_pad = word_tokenizer.pad
        self.unknow = word_tokenizer.unknow
        self.num_workers = num_workers

        all_config = get_config('data_reader/dataset_config.yaml')
        base_config = all_config['BaseReader']
        self.SAMPLE_RATE = base_config['SAMPLE_RATE']
        # there are two special embedding should be initialized here
        self.word_embedding = ChineseTokenEmbedding(
            5000, self.word_tokenizer, model)
        self.pinyin_embedding = PinyinTokenEmbedding(
            self.pinyin_tokenizer, model)

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def audio_to_mel(self, audio):
        audio = self.load_wave(audio, sample_rate=self.SAMPLE_RATE)
        audio = audio.flatten()
        assert audio.shape[-1] < whisper.audio.N_SAMPLES  # or it will be cut
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel

    def get_keys_mask(self, data: dict) -> list[int]:
        return [1 if dataset_keys[i] in data.keys() else 0 for i in range(len(dataset_keys))]

    def feature_encode(self, data: dict, size_of_input: int, mask: bool, task: str, features: Dict[str, list], aim_task: str) -> None:
        tokenizer = self.all_tokenizers[task]
        if task == aim_task:
            if task not in features:
                features[task] = []
                features[task+'_label'] = []
            if mask:
                words = [*tokenizer.sot_task_so_on] + \
                    tokenizer.encode(data[task], default=self.unknow)
            else:
                words = (len(tokenizer.sot_task_so_on) +
                         size_of_input)*[tokenizer.pad_label]
            features[task].append(words)
            features[task+'_label'].append(words[1:]+[tokenizer.eot])

    def __call__(self, input: list[dict]) -> SonicBatch:
        features: Dict[str, list] = {'mel': [], 'mask': []}
        feature_lengths: list[int] = []
        for data in input:
            assert len(set(data.keys()) - set(dataset_keys)
                       ) == 0, 'not all keys in data legal'
            assert 'pinyin' in data.keys() or 'note' in data.keys()
            # 0. update Chinese characters table, and also transcribe traditional to simplified
            self.word_embedding.auto_update(data['hanzi'], self.num_workers)
            # 1. get size of dataset
            size_of_input = len(data['pinyin'] or data['note'])
            feature_lengths.append(
                size_of_input+len(self.pinyin_tokenizer.sot_task_so_on))
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
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'hanzi')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'pinyin')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'note')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'tone')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'slur')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'start')
                self.feature_encode(data, size_of_input,
                                    mask[i], task, features, 'end')

        features['mel'] = torch.concat([m[None, :] for m in features['mel']])
        features['mask'] = torch.Tensor(features['mask']).to(dtype=torch.int32)
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
        [setattr(sonic_batch, field_name, v)
         for field_name, v in features.items()]

        return sonic_batch


@dataclass
class WhisperOfficialBatch:
    mel: Optional[Tensor]
    data: Optional[Tensor]
    data_label: Optional[Tensor]


def whisper_official_batch_to_shape(sonic_batch: WhisperOfficialBatch):
    field_names = [field.name for field in fields(WhisperOfficialBatch)]
    sonic_batch_shape = {}
    for field_name in field_names:
        value = getattr(sonic_batch, field_name)
        if value is not None:
            sonic_batch_shape[field_name] = value.shape
    return sonic_batch_shape


class WhisperOfficialDataCollatorWithPadding:
    def __init__(self, model='tiny', num_workers=1) -> None:
        self.num_workers = num_workers
        all_config = get_config('data_reader/dataset_config.yaml')
        base_config = all_config['BaseReader']
        self.SAMPLE_RATE = base_config['SAMPLE_RATE']
        self.tokenizer = get_tokenizer(
            multilingual=True, language='zh', task='transcribe')
        self.const_pad_label = self.tokenizer.pad
        self.const_pad = self.tokenizer.pad

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def audio_to_mel(self, audio):
        audio = self.load_wave(audio, sample_rate=self.SAMPLE_RATE)
        audio = audio.flatten()
        assert audio.shape[-1] < whisper.audio.N_SAMPLES  # or it will be cut
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel

    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        '''
        Accept input, which is a list of data, containing the keys you want to add to the 
        output data string
        '''
        features: Dict[str, list] = {'mel': [], 'data': [], 'data_label': []}
        feature_lengths: list[int] = []
        for data in input:
            ORDER = 'ORDER' in data and data.pop('ORDER') == 1
            assert len(set(data.keys()) - set(dataset_keys)
                       ) == 0, f'not all keys in data legal. {data.keys()}'
            mel = self.audio_to_mel(data['audio'])
            features['mel'].append(mel)
            keys = list(data.keys())
            keys.remove('audio')
            assert len(keys) >= 1
            for key in keys:
                assert f'<|{key}|>' in self.tokenizer.special_tokens
            # hanzi = data['hanzi']
            # note = data['note']
            data_concated = ''
            for i in range(len(data[keys[0]])):
                # 1. order number, exist when not only hanzi exist
                if ORDER:
                    data_concated += f'<|{i}|>'
                # 2. hanzi
                if 'hanzi' in keys:
                    hanzi = data['hanzi']
                    if hanzi[i] in ['AP', 'SP', 'SL']:
                        hanzi[i] = f'<|{hanzi[i]}|>'
                    data_concated += f'{hanzi[i]}'
                # 3. note
                if 'note' in keys:
                    note = data['note']
                    data_concated += f'<|{note[i]}|>'
            data_concated = self.tokenizer.encode(data_concated, allowed_special="all")
            data_builder = list(self.tokenizer.sot_sequence) 
            data_builder += [
                self.tokenizer.special_tokens[f'<|{key}|>'] for key in keys] 
            if ORDER:
                data_builder += [self.tokenizer.order]
            data_builder += [
                    self.tokenizer.soi] +data_concated + [
                    self.tokenizer.eot]
            features['data'].append(data_builder)
            features['data_label'].append(
                data_builder[:-1]+[self.tokenizer.eot])
            feature_lengths.append(len(data_builder))

        features['mel'] = torch.concat([m[None, :] for m in features['mel']])

        for k, v in features.items():
            if k in ['mel', 'mask']:
                continue
            if 'label' in k:
                constant_values = self.const_pad_label
            else:
                constant_values = self.const_pad
            feature_lengths_ = [len(p) if p is not None else 0 for p in v]
            # print(k, v)
            max_feature_len = max(feature_lengths)
            assert feature_lengths_ == feature_lengths, f'{k}, current {feature_lengths_} not equals to {feature_lengths}'
            padded_data = [torch.nn.functional.pad(torch.tensor(f), (0, max_feature_len - f_len), value=constant_values)
                           for f, f_len in zip(v, feature_lengths)]
            features[k] = torch.stack(padded_data, dim=0)

        sonic_batch = WhisperOfficialBatch(
            mel=features['mel'],
            data=features['data'],
            data_label=features['data_label']
        )

        return sonic_batch


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_with_weights):
        def expand(x):
            if type(x) == tuple and len(x) == 2:
                return (x[0], x[1], '')
            if type(x) == tuple and len(x) == 1:
                return (x[1], 1, '')
            if type(x) != tuple:
                return (x, 1, '')
            else:
                return x
        datasets_with_weights = [expand(x) for x in datasets_with_weights]
        self.datasets = [ds for ds, weight, settings in datasets_with_weights for _ in range(weight)]
        self.settings:list[str] = [settings for ds, weight, settings in datasets_with_weights for _ in range(weight)]
        # self.weights = [weight for _, weight in datasets_with_weights]
        self.lengths = [len(ds) for ds in self.datasets]
        self.cum_lengths = torch.tensor(self.lengths).cumsum(dim=0).tolist()
        # print(self.lengths, self.cum_lengths)
        # total_weight = sum(self.weights)
        # self.weights = [w / total_weight for w in self.weights]
        # self.cum_weights = torch.tensor(self.weights).cumsum(dim=0).tolist()

    def __getitem__(self, index) -> dict:
        dataset_idx, inside_data_idx = self._get_random_dataset_index(index)
        selected_dataset:BaseReader = self.datasets[dataset_idx]
        item = selected_dataset[inside_data_idx]
        if 'order' in self.settings[dataset_idx]:
            item['ORDER'] = 1
        return item

    def _get_random_dataset_index(self, index):
        for i, cum_lengths in enumerate(self.cum_lengths):
            if index < cum_lengths:
                dataset_index = i
                inside_data_idx = index - (cum_lengths - self.lengths[i])
                return dataset_index, inside_data_idx
        
    def __len__(self):
        return sum(self.lengths)
    
if __name__ == '__main__':
    from data_reader.opencpop import OpenCpop
    from data_reader.openslr_33 import Openslr33
    from data_reader.openslr_47 import Openslr47
    '''
    dataset_a = OpenCpop(train=False)
    dataset_b = Openslr33(train=False)
    dataset_c = Openslr47(train=False)
    print(len(dataset_a), len(dataset_b), len(dataset_c), flush=True)
    weighted_dataset = WeightedDataset([(dataset_a, 3), (dataset_b, 3), (dataset_c, 3)])
    print(len(weighted_dataset))
    # exit()
    
    loader = DataLoader(weighted_dataset, 
                            batch_size=2,
                            shuffle=True,
                            collate_fn=WhisperDataCollatorWithPadding(),
                            num_workers=2,
                            persistent_workers=True,
                            drop_last=True,
                          )
    for b in loader:
        # print(b)
        print(sonic_batch_to_shape(b))
        break
    '''
    dataset_a = OpenCpop(train=False, key_filter=['audio', 'hanzi', 'note'])
    # dataset_b = Openslr33(train=False, key_filter=['audio', 'hanzi'])
    dataset_b = OpenCpop(train=False, key_filter=['audio', 'hanzi'])
    dataset_c = OpenCpop(train=False, key_filter=['audio', 'note'])
    weighted_dataset = WeightedDataset([dataset_a, (dataset_a, 1, 'order'), (dataset_b, 1, 'order'), (dataset_c, 1, 'order')])
    collate_fn = WhisperOfficialDataCollatorWithPadding()
    batch_size=8
    loader = DataLoader(weighted_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=2,
                        persistent_workers=True,
                        drop_last=True,
                        )
    for b in loader:
        b: WhisperOfficialBatch = b
        # print(b)
        print(whisper_official_batch_to_shape(b))
        for i in range(batch_size):
            print(collate_fn.tokenizer.decode(
                b.data[i].tolist(), stop_at=collate_fn.tokenizer.eot))
        break
