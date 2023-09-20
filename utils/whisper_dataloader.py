import random
from typing import Any
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch import Tensor
import torchaudio
from data_reader.base_reader import BaseReader
from data_reader.openslr_33 import Openslr33
from utils.general_dataloader import WhisperOfficialBatch, WhisperOfficialDataCollatorWithPadding, whisper_official_batch_to_shape, dataset_keys
from utils.load_checkpoint import get_config
from utils.naive_tokenizer import get_tokenizer
import torchaudio.transforms as at
import whisper
from torchaudio.functional import vad

'''
only for opencpop
'''
class OpenCpopDataCollatorWithPadding(WhisperOfficialDataCollatorWithPadding):
    def __init__(self, model='tiny', num_workers=1, auto_merge_tensor=True) -> None:
        super().__init__(model, num_workers)
        self.auto_merge_tensor = auto_merge_tensor
    
    def add_random_numbers(self, array, n:int=0, k:int=0):
        lower_bound = torch.min(array).item() / 1000
        upper_bound = torch.max(array).item() / 1000
        array_with_random_numbers = array
        if n > 0:
            random_numbers_front = torch.randn(1, n) * (upper_bound - lower_bound) + lower_bound
            array_with_random_numbers = torch.cat((random_numbers_front, array_with_random_numbers), dim=-1)
        if k > 0:
            random_numbers_back = torch.randn(1, k) * (upper_bound - lower_bound) + lower_bound
            array_with_random_numbers = torch.cat((array_with_random_numbers, random_numbers_back), dim=-1)
        return array_with_random_numbers

    '''
    add_first/add_last works by add noise before and after audio by seconds
    '''
    def audio_to_mel(self, audio, add_first=.0, add_last=.0):
        if add_first > .0 or add_last > .0:
            assert len(audio.shape) == 2, f'shape of audio should be [1, N], but {audio.shape} found'
            add_first = int(self.SAMPLE_RATE*add_first)
            add_last = int(self.SAMPLE_RATE*add_last)
            audio = self.add_random_numbers(audio, add_first, add_last)
            if audio.shape[-1] >= whisper.audio.N_SAMPLES:
                print(f"warn, audio len={audio.shape[-1]} > N_SAMPLES={whisper.audio.N_SAMPLES}")
        audio_flat = audio.flatten()
        audio_pad = whisper.pad_or_trim(audio_flat)
        mel = whisper.log_mel_spectrogram(audio_pad)
        return mel, audio

    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        '''
        Accept input, which is a list of data, containing the keys you want to add to the 
        output data string
        '''
        features: dict[str, list] = {'mel': [], 'data': [], 'data_label': [], 'waveform': []}
        feature_lengths: list[int] = []
        for data in input:
            if 'waveform' not in data:
                waveform = self.load_wave(data['audio'], sample_rate=self.SAMPLE_RATE)
                data['waveform'] = waveform
            else:
                waveform = data['waveform']
            keys = list(data.keys())
            add_first = .0
            add_last = .0
            if 'pad' in keys:
                add_first = random.random() * 0.9
                add_last = random.random() * 0.9
            # audio = self.load_wave(data['audio'], sample_rate=self.SAMPLE_RATE)
            mel, waveform = self.audio_to_mel(waveform, add_first, add_last)
            features['mel'].append(mel)
            features['waveform'].append(waveform)
            keys.remove('audio')
            assert len(keys) >= 1
            data_concated = ''
            notimestamp = 'notimestamp' in keys
            for i in range(len(data[keys[0]])):
                # 0. start
                if 'start' in keys and not notimestamp:
                    start = data['start']
                    value = (float(start[i]) + add_first + 0.01) // 0.02
                    data_concated += f'<|{value * 0.02:.2f}|>'
                # 1. order number, exist when not only hanzi exist
                if 'order' in keys:
                    data_concated += f'<|{i}|>'
                # 2. hanzi
                if 'hanzi' in keys:
                    hanzi = data['hanzi']
                    if hanzi[i] in ['AP', 'SP', 'SL']:
                        hanzi[i] = f'<|{hanzi[i]}|><|{hanzi[i]}|>' 
                        # duplate spaticial tokens which will be on the same place with hanzi because hanzi takes 2 tokens
                    data_concated += f'{hanzi[i]}'
                # 3. note
                if 'note' in keys:
                    note = data['note']
                    data_concated += f'<|{note[i]}|>'
                # 4. end (of timestamp)
                if 'end' in keys and not notimestamp:
                    end = data['end']
                    value = (float(end[i]) + add_first + 0.01) // 0.02
                    data_concated += f'<|{value * 0.02:.2f}|>'
            data_concated = self.tokenizer.encode(data_concated, allowed_special="all")
            data_builder = [self.tokenizer.soi, self.tokenizer.singing]
            if 'order' in keys:
                data_builder += [self.tokenizer.order]
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps]
            data_builder += list(self.tokenizer.sot_sequence)
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps] # this duplication is needed
            data_builder += data_concated + [self.tokenizer.eot]
            features['data'].append(data_builder)
            features['data_label'].append(
                data_builder[1:]+[self.tokenizer.pad])
            feature_lengths.append(len(data_builder))

        if self.auto_merge_tensor:
            features['mel'] = torch.concat([m[None, :] for m in features['mel']])
            for k, v in features.items():
                if k in ['mel', 'mask', 'waveform']:
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
            data_label=features['data_label'],
            waveform=features['waveform']
        )

        return sonic_batch
 
'''
for all speech dataset
'''   
class SpeechDataCollatorWithPadding(OpenCpopDataCollatorWithPadding):
    def __init__(self, model='tiny', num_workers=1, num_concat=None, auto_merge_tensor=True) -> None:
        super().__init__(model, num_workers)
        self.num_concat = num_concat
        self.auto_merge_tensor = auto_merge_tensor
        
    def random_concat_input(self, input: list[dict], idx:int, random_concat_num:int) -> tuple[list[int], Tensor]:
        shuffled_input = input[:]
        first_data = shuffled_input[idx]
        del shuffled_input[idx]
        random.shuffle(shuffled_input)
        shuffled_input = [first_data] + shuffled_input
        data_concated = ''
        time_accumulated = 0
        wav_concated = []
        order_count = 0
        def finished_concat():
            data_concated_encoded = self.tokenizer.encode(data_concated, allowed_special="all")
            data_builder = []
            if 'order' in keys:
                data_builder += [self.tokenizer.soi]
                data_builder += [self.tokenizer.order]
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps]
            data_builder += list(self.tokenizer.sot_sequence)
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps] # this duplication is needed
            data_builder += data_concated_encoded + [self.tokenizer.eot]
            wav_concated_tensor = torch.cat(wav_concated, dim=-1)
            # input[idx]['waveform'] = wav_concated_tensor
            mel, _ = self.audio_to_mel(wav_concated_tensor)
            return data_builder, mel, wav_concated_tensor
            
        for batch_idx in range(random_concat_num):
            data = shuffled_input[batch_idx]
            keys = list(data.keys())
            add_first = .0
            if 'pad' in keys:
                add_first = random.random() * 0.9
            waveform = data['waveform']
            # time_len, which is the actual length of the input waveform
            wav_len = waveform.shape[-1]
            time_len = float(wav_len) / self.SAMPLE_RATE
            # append noise before waveform
            add_first_samples = int(self.SAMPLE_RATE*add_first)
            waveform = self.add_random_numbers(waveform, add_first_samples, 0)
           
            # if this data will cause time > 30s
            if time_accumulated + time_len + add_first > 29.9:
                return finished_concat()
            wav_concated.append(waveform)
            keys.remove('audio')
            assert len(keys) >= 1
            notimestamp = 'notimestamp' in keys
            # 0. start
            if not notimestamp:
                time_accumulated += add_first
                value = (float(time_accumulated) + 0.01) // 0.02
                data_concated += f'<|{value * 0.02:.2f}|>'
            if 'hanzi' in keys and batch_idx % self.num_concat != 0 and notimestamp:
                if 'order' in keys:
                    data_concated += f'<|{order_count}|><|SP|><|SP|>'
                    order_count += 1
                else:
                    data_concated += '，'
            for i in range(len(data[keys[0]])):
                # 0. order
                if 'order' in keys:
                    data_concated += f'<|{order_count}|>'
                    order_count += 1
                # 1. hanzi
                if 'hanzi' in keys:
                    hanzi = data['hanzi']
                    if hanzi[i] in ['AP', 'SP', 'SL']:
                        hanzi[i] = f'<|{hanzi[i]}|><|{hanzi[i]}|>'
                    data_concated += hanzi[i]
            # 2. end (of timestamp)
            if not notimestamp:
                time_accumulated += time_len
                value = (float(time_accumulated) + 0.01) // 0.02
                data_concated += f'<|{value * 0.02:.2f}|>'
        return finished_concat()
    
    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        features: dict[str, list] = {'mel': [], 'data': [], 'data_label': [], 'waveform': []}
        feature_lengths: list[int] = []
        input_size = len(input)
        if self.num_concat == None:
            self.num_concat = input_size
            
        for data in input:
            if 'waveform' in data:
                continue
            waveform = self.load_wave(data['audio'], sample_rate=self.SAMPLE_RATE)
            # # trim silence and quiet background sounds
            # vad_waveform = vad(waveform, sample_rate=self.SAMPLE_RATE)
            # reversed_waveform = torch.flip(vad_waveform, [1])
            # reversed_vad_waveform = vad(reversed_waveform, sample_rate=self.SAMPLE_RATE)
            # waveform = torch.flip(reversed_vad_waveform, [1])
            data['waveform'] = waveform
            
        for i in range(input_size):
            data_builder, mel, waveform_concat = self.random_concat_input(input, i, random.randint(1, min(input_size, self.num_concat)))
            features['data'].append(data_builder)
            features['data_label'].append(
                data_builder[1:]+[self.tokenizer.pad])
            feature_lengths.append(len(data_builder))
            features['mel'].append(mel)
            features['waveform'].append(waveform_concat)

        if self.auto_merge_tensor:
            features['mel'] = torch.concat([m[None, :] for m in features['mel']])
            for k, v in features.items():
                if k in ['mel', 'mask', 'waveform']:
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
            data_label=features['data_label'],
            waveform=features['waveform']
        )
        return sonic_batch
    
class mixedDataCollator(WhisperOfficialDataCollatorWithPadding):
    def __init__(self, collators: dict[str: WhisperOfficialDataCollatorWithPadding]) -> None:
        super().__init__()
        self.collators = collators
        self.collator_names = collators.keys()
        
    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        features: dict[str, list] = {'mel': [], 'data': [], 'data_label': [], 'waveform': [], 'clusters': []}
        feature_lengths: list[int] = []
        data_clusters:dict[str,list(dict)] = dict()
        for data in input:
            assert data['cluster'] in self.collator_names
            if data['cluster'] not in data_clusters:
                data_clusters[data['cluster']] = []
            data_clusters[data['cluster']].append(data)
        # print(data_clusters)
        for k, v in data_clusters.items():
            v: list(dict) = v
            collator:WhisperOfficialDataCollatorWithPadding = self.collators[k]
            batch = collator(v)
            features['clusters'].extend([k]*len(batch.data))
            features['mel'].extend(batch.mel)
            features['data'].extend(batch.data)
            feature_lengths.extend([len(data) for data in batch.data])
            features['data_label'].extend(batch.data_label)
            features['waveform'].extend(batch.waveform)
            
        features['mel'] = torch.concat([m[None, :] for m in features['mel']])
        for k, v in features.items():
            if k in ['mel', 'mask', 'waveform', 'clusters']:
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
            data_label=features['data_label'],
            waveform=features['waveform'],
            clusters=features['clusters']
        )
        return sonic_batch

'''
This dataset builder can concate multiple dataset into on.
you may duplicate some of datasets by set the second value in tuple to more than 1,
and the 3rd in tuple is the settings. These settings will be used in collator.
notimestamp: no <|0.12|> in openslr
pad: add noise before the audio
order: add <|0|> in each hanzi
cluster:XXX for mixedDataCollator
=== Example ===
wrapped_dataset = WrappedDataset([
    (openslr, 1, ['notimestamp', 'pad']),
    (openslr, 1, ['notimestamp']),
    (openslr, 1, ['notimestamp', 'cluster:speech']),
    (openCpop, 1, ['order', 'pad', 'cluster:opencpop']),
    (openCpop, 1, ['pad', 'cluster:opencpop2']),
    ])
'''
class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_with_settings: list[tuple[BaseReader, int, list[str]]]):
        self.datasets:list[BaseReader] = [ds for ds, weight, settings in datasets_with_settings for _ in range(weight)]
        self.settings:list[str] = [settings for ds, weight, settings in datasets_with_settings for _ in range(weight)]
        self.lengths = [len(ds) for ds in self.datasets]
        self.cum_lengths = torch.tensor(self.lengths).cumsum(dim=0).tolist()

    def __getitem__(self, index) -> dict:
        dataset_idx, inside_data_idx = self._get_random_dataset_index(index)
        selected_dataset:BaseReader = self.datasets[dataset_idx]
        item = selected_dataset[inside_data_idx]
        for key in self.settings[dataset_idx]:
            if ':' in key:
                item[key.split(':')[0]] = key.split(':')[1]
            item[key] = 1
        return item
    
    def _get_random_dataset_index(self, index):
        for i, cum_lengths in enumerate(self.cum_lengths):
            if index < cum_lengths:
                dataset_index = i
                inside_data_idx = index - (cum_lengths - self.lengths[i])
                return dataset_index, inside_data_idx
        
    def __len__(self):
        return sum(self.lengths)
    
'''
This dataset_builder has 1 main dataset, which should be the largest dataset and it should 
be at least 100 times larger than other datasets.
Then other dataset can replace this main dataset by the fraction as defined.
'''
class RandomReplaceDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_with_settings: list[tuple[BaseReader, int, list[str]]]):
        self.datasets:list[BaseReader] = [ds for ds, weights, settings in datasets_with_settings]
        self.weights:list[str] = [weights for ds, weights, settings in datasets_with_settings]
        self.settings:list[str] = [settings for ds, weights, settings in datasets_with_settings]
        self.lengths = [len(ds) for ds in self.datasets]
        assert self.lengths[0] == max(self.lengths)
        self.cum_weights = torch.tensor(self.weights).cumsum(dim=0).tolist()

    def __getitem__(self, index) -> dict:
        dataset_idx = random.choices(range(len(self.weights)), weights=self.weights, k=1)[0]
        selected_dataset:BaseReader = self.datasets[dataset_idx]
        item = selected_dataset[index % self.lengths[dataset_idx]]
        for key in self.settings[dataset_idx]:
            if ':' in key:
                item[key.split(':')[0]] = key.split(':')[1]
            item[key] = 1
        return item
        
    def __len__(self):
        return max(self.lengths)

if __name__ == '__main__':
    from data_reader.opencpop import OpenCpop
    def test_opencpop():
        openCpop = OpenCpop(train=False, key_filter=['audio', 'hanzi', 'note', 'start', 'end', 'waveform'])
        wrapped_dataset = WrappedDataset([
            # (openCpop, 1, ['order', 'notimestamp']),
            (openCpop, 1, ['order', 'pad']),
            # (openCpop, 1, ['pad'])
            ])
        collate_fn = OpenCpopDataCollatorWithPadding()
        batch_size=8
        loader = DataLoader(wrapped_dataset,
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
                print('Data', collate_fn.tokenizer.decode(
                    b.data[i].tolist(), stop_at=collate_fn.tokenizer.eot))
                torchaudio.save('logs/test_vad.wav', b.waveform[i], openCpop.SAMPLE_RATE)
                # print('Labl',collate_fn.tokenizer.decode(
                #     b.data_label[i].tolist(), stop_at=collate_fn.tokenizer.eot))
            break
        
    def test_speech():
        openslr = Openslr33(train=False, key_filter=['audio', 'hanzi', 'waveform'])
        wrapped_dataset = WrappedDataset([
            # (openslr, 1, ['notimestamp', 'pad']),
            # (openslr, 1, ['notimestamp']),
            (openslr, 1, ['pad'])
            ])
        collate_fn = SpeechDataCollatorWithPadding()
        batch_size=8
        loader = DataLoader(wrapped_dataset,
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
                print('Data', i, collate_fn.tokenizer.decode(
                    b.data[i].tolist(), stop_at=collate_fn.tokenizer.eot))
                print('Labl', i, collate_fn.tokenizer.decode(
                    b.data_label[i].tolist(), stop_at=collate_fn.tokenizer.eot))
                print('Mel', i, b.mel[i].shape)
                torchaudio.save('logs/test_vad.wav', b.waveform[i], openslr.SAMPLE_RATE)
                break
            break
        
    def test_mix():
        openslr = Openslr33(train=False, key_filter=['audio', 'hanzi', 'waveform'])
        openCpop = OpenCpop(train=False, key_filter=['audio', 'hanzi', 'note', 'start', 'end', 'waveform'])
        wrapped_dataset = RandomReplaceDataset([
            # (openslr, 1, ['notimestamp', 'pad']),
            (openslr, 0.3, ['notimestamp', 'order', 'cluster:speech2']),
            (openslr, 0.3, ['notimestamp', 'cluster:speech']),
            (openCpop, 0.2, ['order', 'pad', 'cluster:opencpop']),
            (openCpop, 0.2, ['pad', 'cluster:opencpop2']),
            ])
        speech_collate_fn = SpeechDataCollatorWithPadding(auto_merge_tensor=False)
        opencpop_collate_fn = OpenCpopDataCollatorWithPadding(auto_merge_tensor=False)
        collate_fn = mixedDataCollator({
            'speech': speech_collate_fn,
            'speech2': speech_collate_fn,
            'opencpop': opencpop_collate_fn,
            'opencpop2': opencpop_collate_fn
        })
        batch_size=10
        loader = DataLoader(wrapped_dataset,
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
                print('clusters', b.clusters[i])
                print('Data', i, collate_fn.tokenizer.decode(
                    b.data[i].tolist(), stop_at=collate_fn.tokenizer.eot))
                print('Labl', i, collate_fn.tokenizer.decode(
                    b.data_label[i].tolist(), stop_at=collate_fn.tokenizer.eot))
                print('Mel', i, b.mel[i].shape)
                torchaudio.save('logs/test_vad.wav', b.waveform[i], openslr.SAMPLE_RATE)
                input("enter for next")
                # break
            break

    # test_opencpop()
    # test_speech()
    test_mix()