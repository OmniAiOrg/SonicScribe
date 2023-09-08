import random
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
    def __init__(self, model='tiny', num_workers=1) -> None:
        super().__init__(model, num_workers)
    
    def add_random_numbers(self, array, n, k):
        lower_bound = torch.min(array).item() / 100
        upper_bound = torch.max(array).item() / 100
        random_numbers_front = torch.randn(n) * (upper_bound - lower_bound) + lower_bound
        random_numbers_back = torch.randn(k) * (upper_bound - lower_bound) + lower_bound
        array_with_random_numbers = torch.cat((random_numbers_front, array, random_numbers_back))
        return array_with_random_numbers

    '''
    add_first/add_last works by add noise before and after audio by seconds
    '''
    def audio_to_mel(self, audio, add_first=.0, add_last=.0):        
        audio = audio.flatten()
        add_first = int(self.SAMPLE_RATE*add_first)
        add_last = int(self.SAMPLE_RATE*add_last)
        audio = self.add_random_numbers(audio, add_first, add_last)
        assert len(audio.shape) == 1
        assert audio.shape[-1] < whisper.audio.N_SAMPLES  # or it will be cut
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel

    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        '''
        Accept input, which is a list of data, containing the keys you want to add to the 
        output data string
        '''
        features: dict[str, list] = {'mel': [], 'data': [], 'data_label': []}
        feature_lengths: list[int] = []
        for data in input:
            keys = list(data.keys())
            add_first = .0
            add_last = .0
            if 'pad' in keys:
                add_first = random.random() * 0.9
                add_last = random.random() * 0.9
            audio = self.load_wave(data['audio'], sample_rate=self.SAMPLE_RATE)
            mel = self.audio_to_mel(audio, add_first, add_last)
            features['mel'].append(mel)
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
                        hanzi[i] = f'<|{hanzi[i]}|>'
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
            data_builder = [self.tokenizer.sot_prev, self.tokenizer.singing]
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
 
'''
for all speech dataset
'''   
class SpeechDataCollatorWithPadding(OpenCpopDataCollatorWithPadding):
    def __init__(self, model='tiny', num_workers=1, num_concat=None) -> None:
        super().__init__(model, num_workers)
        self.num_concat = num_concat
        
    def random_concat_input(self, input: list[dict], idx:int, random_concat_num:int) -> tuple[list[int], Tensor]:
        shuffled_input = input[:]
        first_data = shuffled_input[idx]
        del shuffled_input[idx]
        random.shuffle(shuffled_input)
        shuffled_input = [first_data] + shuffled_input
        data_concated = ''
        time_accumulated = 0
        wav_concated = []
        def finished_concat():
            data_concated_encoded = self.tokenizer.encode(data_concated, allowed_special="all")
            data_builder = []
            if 'order' in keys:
                data_builder += [self.tokenizer.sot_prev]
            if 'order' in keys:
                data_builder += [self.tokenizer.order]
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps]
            data_builder += list(self.tokenizer.sot_sequence)
            if notimestamp:
                data_builder += [self.tokenizer.no_timestamps] # this duplication is needed
            data_builder += data_concated_encoded + [self.tokenizer.eot]
            wav_concated_tensor = torch.cat(wav_concated, dim=-1)
            mel = self.audio_to_mel(wav_concated_tensor, add_first, add_last)
            return data_builder, mel
            
        for batch_idx in range(random_concat_num):
            data = shuffled_input[batch_idx]
            keys = list(data.keys())
            add_first = .0
            add_last = .0
            if 'pad' in keys:
                add_first = random.random() * 0.9
                add_last = random.random() * 0.9
            waveform = data['waveform']
            # get timestamp
            wav_len = waveform.shape[-1]
            time_len = float(wav_len) / self.SAMPLE_RATE
            # if this data will cause time > 30s
            if time_accumulated + time_len + add_first + add_last > 29.9:
                return finished_concat()
            wav_concated.append(waveform)
            keys.remove('audio')
            assert len(keys) >= 1
            notimestamp = 'notimestamp' in keys
            # 0. start
            if not notimestamp:
                time_accumulated += add_first
                start = time_accumulated
                value = (float(start) + 0.01) // 0.02
                data_concated += f'<|{value * 0.02:.2f}|>'
            if 'hanzi' in keys and batch_idx % self.num_concat != 0 and notimestamp:
                data_concated += '，'
            for i in range(len(data[keys[0]])):
                # 1. hanzi
                if 'hanzi' in keys:
                    hanzi = data['hanzi']
                    if hanzi[i] in ['AP', 'SP', 'SL']:
                        hanzi[i] = f'<|{hanzi[i]}|>'
                    data_concated += f'{hanzi[i]}'
            # 2. end (of timestamp)
            if not notimestamp:
                time_accumulated += time_len
                end = time_accumulated
                time_accumulated += add_last
                value = (float(end) + 0.01) // 0.02
                data_concated += f'<|{value * 0.02:.2f}|>'
        return finished_concat()
    
    def __call__(self, input: list[dict]) -> WhisperOfficialBatch:
        features: dict[str, list] = {'mel': [], 'data': [], 'data_label': []}
        feature_lengths: list[int] = []
        input_size = len(input)
        if self.num_concat == None:
            self.num_concat = input_size
            
        for data in input:
            if 'waveform' in data:
                continue
            waveform = self.load_wave(data['audio'], sample_rate=self.SAMPLE_RATE)
            # trim silence and quiet background sounds
            vad_waveform = vad(waveform, sample_rate=self.SAMPLE_RATE)
            reversed_waveform = torch.flip(vad_waveform, [1])
            reversed_vad_waveform = vad(reversed_waveform, sample_rate=self.SAMPLE_RATE)
            waveform = torch.flip(reversed_vad_waveform, [1])
            data['waveform'] = waveform
            
        for i in range(input_size):
            data_builder, mel = self.random_concat_input(input, i, random.randint(1, self.num_concat))
            features['data'].append(data_builder)
            features['data_label'].append(
                data_builder[1:]+[self.tokenizer.pad])
            feature_lengths.append(len(data_builder))
            features['mel'].append(mel)
        
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

if __name__ == '__main__':
    def test_opencpop():
        from data_reader.opencpop import OpenCpop
        openCpop = OpenCpop(train=False, key_filter=['audio', 'hanzi', 'note', 'start', 'end'])
        wrapped_dataset = WrappedDataset([
            (openCpop, 1, ['order', 'notimestamp']),
            (openCpop, 1, ['order', 'pad']),
            (openCpop, 1, ['pad'])
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
            break

    # test_opencpop()
    test_speech()