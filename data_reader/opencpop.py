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
from utils.ph import get_initials_and_finals
from utils.naive_tokenizer import NaiveTokenizer
from utils.note import notes
from utils.general_dataloader import SonicData
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer

class OpenCpop(BaseReader):
    def __init__(self, train=True) -> None:
        super().__init__(train)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp'
        self.audio_transcript_pair_list
        self.NONE_TEXT = str(self.config['NONE_TEXT'])
        self.SLUR_TEXT = str(self.config['SLUR_TEXT'])
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.audio_transcript_pair_list = self.get_dataset(train)
        initials, finals = get_initials_and_finals()
        self.initials_tokenizer = NaiveTokenizer(initials+[self.NONE_TEXT, self.SLUR_TEXT]+['AP','SP'], '<|initial|>')
        self.finals_tokenizer = NaiveTokenizer(finals+['AP','SP'], '<|final|>')
        self.note_tokenizer = NaiveTokenizer(notes, '<|note|>')
        self.slur_tokenizer = NaiveTokenizer(['0', '1', '2'], '<|slur|>')
        self.word_tokenizer = get_tokenizer(True, language='zh', task='transcribe')
        
    def get_dataset(self, train):
        dataset_txt = 'train' if train else 'test'
        data = self.parse_txt(self.path + f'/{dataset_txt}.txt')
        return self.get_audio_file_list(data, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'{dataset_txt}_audio_opencpop')
    
    def parse_line(self, id, text, phoneme, note, note_duration, phoneme_duration, slur):
        none_text = self.NONE_TEXT
        slur_text = self.SLUR_TEXT
        # 在此修改格式，使满足元辅音区分后共同预测，简化预测过程
        initials, finals = get_initials_and_finals()
        phoneme = phoneme.split(' ')
        note = note.split(' ')
        note_duration = [float(var) for var in note_duration.split(' ')]
        org_slur = slur
        slur = slur.split('\n')[0].split(' ')
        hanzi_initials = []
        hanzi_finals = []
        hanzi_note = []
        hanzi_note_duration = []
        hanzi_slur = []
        for i in range(len(phoneme)):
            p = phoneme[i]
            assert slur[i] in ['0', '1'], f'slur[i]={slur[i]} not in 0/1, org_slur={org_slur}'
            if p not in initials and p not in finals:
                # p in ['AP', 'SP']
                hanzi_initials.append(p)
                hanzi_finals.append(p)
            elif p in initials:
                # 元音就一定有辅音，然后跳过等到辅音存储
                assert i != len(phoneme)-1 and phoneme[i+1] in finals
                continue
            elif p in finals:
                # 辅音考虑以下情况
                # 1. 前一个是元音的辅音，不slur，这就是一般汉字“sh ui”
                # 2. 前一个是元音的辅音，slur，不应该存在
                # 3. 前一个是辅音的辅音，不slur，比如“好啊”
                # 4. 前一个是辅音的辅音，slur，也就是延音
                # 5. 前一个都不是的辅音，不slur，“AP 啊”
                # 6. 前一个都不是的辅音，slur，不应该存在
                if phoneme[i-1] in initials:
                    assert slur[i] == '0', 'slur[i] can only be 0'
                    assert note[i] == note[i-1]
                    assert note_duration[i] == note_duration[i-1]
                    hanzi_initials.append(phoneme[i-1])
                    hanzi_finals.append(p)
                elif phoneme[i-1] in finals:
                    if slur[i] == '0':
                        hanzi_initials.append(none_text)
                        hanzi_finals.append(p)
                    elif slur[i] == '1':
                        hanzi_initials.append(slur_text)
                        hanzi_finals.append(p)
                else:
                    assert phoneme[i-1] in ['AP', 'SP'], f'phoneme[i-1]={phoneme[i-1]} not in AP/SP'
                    assert slur[i] == '0', 'slur[i] can only be 0'
                    hanzi_initials.append(none_text)
                    hanzi_finals.append(p)
            hanzi_note.append(note[i])
            hanzi_note_duration.append(note_duration[i])
            hanzi_slur.append(slur[i])
        return id, text, hanzi_initials, hanzi_finals, hanzi_note, hanzi_note_duration, hanzi_slur
                
    def parse_txt(self, file_name):
        data = []
        with open(file_name, 'r') as f:
            for line in f:
                if len(line) < 10:
                    continue
                id, text, phoneme, note, note_duration, phoneme_duration, slur = line.split('|')
                id, text, initials, finals, note, note_duration, slur = self.parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
                data.append((id, text, initials, finals, note, note_duration, slur))
                # text not matter, just dor debug
        return data
        
    def get_audio_file_list(self, data, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000, save_name = None):
        save_dir = self.pickle_path+'/'+save_name
        if save_name is not None:
            if os.path.isfile(save_dir):
                with open(save_dir,'rb') as out_data:
                    return pickle.load(out_data)
        print('Because this is the first time you read this dataset, please wait for data index')
        audio_transcript_pair_list = []
        # initials, finals = get_initials_and_finals()
        for (id, text, initials, finals, note, note_duration, slur) in tqdm(data):
            # check whether audio exist and legal
            audio_dir = self.path+'/wavs/'+id+'.wav'
            if not os.path.isfile(audio_dir):
                print(f"{audio_dir} not exist")
                continue
            audio = self.load_wave(audio_dir, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print(audio_dir, len(text), len(audio))
                continue
            audio_transcript_pair_list.append((str(audio_dir), text, initials, finals, note, note_duration, slur))
            if save_name is not None:
                if not os.path.exists(self.pickle_path):
                    os.makedirs(self.pickle_path)
                with open(save_dir,'wb') as in_data:
                    pickle.dump(audio_transcript_pair_list,in_data,pickle.HIGHEST_PROTOCOL)
        return audio_transcript_pair_list
        
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        audio_dir, text, initials, finals, note, note_duration, slur = pair

        # audio
        audio = self.load_wave(audio_dir, sample_rate=self.SAMPLE_RATE)
        audio = audio.flatten()
        assert audio.shape[-1] < whisper.audio.N_SAMPLES # or it will be cut
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        num_sot = [0]*len([*self.note_tokenizer.sot_sequence_including_notimestamps])

        initials = [*self.initials_tokenizer.sot_sequence_including_notimestamps] + self.initials_tokenizer.encode(initials)
        finals = [*self.finals_tokenizer.sot_sequence_including_notimestamps] + self.finals_tokenizer.encode(finals)
        
        note = [*self.note_tokenizer.sot_sequence_including_notimestamps] + self.note_tokenizer.encode(note)
        note_duration = num_sot + note_duration
        slur = [*self.slur_tokenizer.sot_sequence_including_notimestamps] + self.slur_tokenizer.encode(slur)

        initials_label = initials[1:] + [self.initials_tokenizer.eot]
        finals_label = finals[1:] + [self.finals_tokenizer.eot]
        note_label = note[1:] + [self.note_tokenizer.eot]
        note_duration_label = note_duration[1:] + [0]
        slur_label = slur[1:] + [self.slur_tokenizer.eot]


        # text, phoneme, note, note_duration, slur
        return SonicData(
            mel, 
            words=[word for word in text],
            original_text=text,
            initials=initials,
            initials_label=initials_label,
            finals=finals,
            finals_label=finals_label,
            note=note,
            note_label=note_label,
            note_duration=note_duration,
            note_duration_label=note_duration_label,
            slur=slur,
            slur_label=slur_label,
        )
        
    
    

if __name__ == '__main__':
    oc_train = OpenCpop()
    oc_test = OpenCpop(train=False)
    print(oc_test[0])
    