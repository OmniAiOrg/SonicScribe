# https://www.openslr.org/33

import dataclasses
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
from utils.tokenizers import *

class OpenCpop(BaseReader):
    def __init__(self, train=True) -> None:
        super().__init__(train)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp_v3'
        self.audio_transcript_pair_list
        self.NONE_TEXT = str(self.config['NONE_TEXT'])
        self.SLUR_TEXT = str(self.config['SLUR_TEXT'])
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.audio_transcript_pair_list = self.get_dataset(train)
        
        self.initials_tokenizer = initials_tokenizer
        self.finals_tokenizer = finals_tokenizer
        self.note_tokenizer = note_tokenizer
        self.slur_tokenizer = slur_tokenizer
        self.word_tokenizer = words_tokenizer
        
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
        hanzi_words = []
        
        hanzi_words_idx = 0
        for i in range(len(phoneme)):
            p = phoneme[i]
            assert slur[i] in ['0', '1'], f'slur[i]={slur[i]} not in 0/1, org_slur={org_slur}'
            if p not in initials and p not in finals:
                # p in ['AP', 'SP']
                hanzi_initials.append(p)
                hanzi_finals.append(p)
                hanzi_words.append(p)
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
                if phoneme[i-1] in initials: # 1
                    assert slur[i] == '0', 'slur[i] can only be 0'
                    assert note[i] == note[i-1]
                    assert note_duration[i] == note_duration[i-1]
                    hanzi_initials.append(phoneme[i-1])
                    hanzi_finals.append(p)
                    hanzi_words.append(text[hanzi_words_idx])
                    hanzi_words_idx += 1
                elif phoneme[i-1] in finals:
                    if slur[i] == '0': 
                        hanzi_initials.append(none_text)
                        hanzi_finals.append(p)
                        hanzi_words.append(text[hanzi_words_idx])
                        hanzi_words_idx += 1
                    elif slur[i] == '1': 
                        hanzi_initials.append(slur_text)
                        hanzi_finals.append(p)
                        hanzi_words.append('SL')
                else:
                    assert phoneme[i-1] in ['AP', 'SP'], f'phoneme[i-1]={phoneme[i-1]} not in AP/SP'
                    assert slur[i] == '0', 'slur[i] can only be 0'
                    hanzi_initials.append(none_text)
                    hanzi_finals.append(p)
                    hanzi_words.append(text[hanzi_words_idx])
                    hanzi_words_idx += 1
            hanzi_note.append(note[i])
            hanzi_note_duration.append(note_duration[i])
            hanzi_slur.append(slur[i])
        assert hanzi_words_idx == len(text), f'text has {len(text)} chars, but only {hanzi_words_idx} added. hanzi_words={hanzi_words}, text={text}, hanzi_initials={hanzi_initials}'
        return id, text, hanzi_initials, hanzi_finals, hanzi_note, hanzi_note_duration, hanzi_slur, hanzi_words
                
    def parse_txt(self, file_name):
        data = []
        with open(file_name, 'r') as f:
            for line in f:
                if len(line) < 10:
                    continue
                id, text, phoneme, note, note_duration, phoneme_duration, slur = line.split('|')
                id, text, initials, finals, note, note_duration, slur, hanzi_words = self.parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
                data.append((id, text, initials, finals, note, note_duration, slur, hanzi_words))
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
        for (id, text, initials, finals, note, note_duration, slur, hanzi_words) in tqdm(data):
            # check whether audio exist and legal
            audio_dir = self.path+'/wavs/'+id+'.wav'
            if not os.path.isfile(audio_dir):
                print(f"{audio_dir} not exist")
                continue
            audio = self.load_wave(audio_dir, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print(audio_dir, len(text), len(audio))
                continue
            audio_transcript_pair_list.append((str(audio_dir), text, initials, finals, note, note_duration, slur, hanzi_words))
            if save_name is not None:
                if not os.path.exists(self.pickle_path):
                    os.makedirs(self.pickle_path)
                with open(save_dir,'wb') as in_data:
                    pickle.dump(audio_transcript_pair_list,in_data,pickle.HIGHEST_PROTOCOL)
        return audio_transcript_pair_list
        
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        audio_dir, text, initials, finals, note, note_duration, slur, hanzi_words = pair

        # audio
        if self.dummy_reader:
            audio = np.zeros([1,2])
        else:
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
        # TODO: words 是有问题的
        words= [*self.word_tokenizer.sot_sequence_including_notimestamps] + self.word_tokenizer.encode(hanzi_words)

        # text, phoneme, note, note_duration, slur
        return SonicData(
            mel, 
            words=words,
            original_text=text,
            initials=initials,
            finals=finals,
            note=note,
            note_duration=note_duration,
            slur=slur,
        )
        
    
    

if __name__ == '__main__':
    # oc_train = OpenCpop()
    oc_test = OpenCpop(train=False)
    # field_names = [field.name for field in dataclasses.fields(SonicData)]
    # for fn in field_names:
    #     print(fn, len(getattr(oc_test[0], fn)) if getattr(oc_test[0], fn) is not None else None)
    
    
    line = '2003000102|如果云层是天空的一封信|r u SP g uo uo c eng sh i t ian k ong d e y i f eng x in in SP AP|A4 A4 rest F4 F4 G4 F4 F4 F4 F4 A4 A4 F4 F4 F4 F4 A4 A4 F4 F4 G4 G4 E4 rest rest|0.448380 0.448380 0.101640 0.164300 0.164300 0.343380 0.398490 0.398490 0.312600 0.312600 0.419180 0.419180 0.291000 0.291000 0.166250 0.166250 0.359290 0.359290 0.268160 0.268160 0.449950 0.449950 1.233690 0.325870 0.564150|0.14206 0.30632 0.10164 0.03859 0.12571 0.34338 0.18389 0.2146 0.16473 0.14787 0.11004 0.30914 0.08007 0.21093 0.03995 0.1263 0.11724 0.24205 0.12254 0.14562 0.26 0.18995 1.23369 0.32587 0.56415|0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0'
    id, text, phoneme, note, note_duration, phoneme_duration, slur = line.split('|')
    id, text, initials, finals, note, note_duration, slur, hanzi_words = oc_test.parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
    print(hanzi_words, text, initials)
    
    # TODO: 错误的编码：AssertionError: text has 11 chars, but only 10 added. hanzi_words=['如', 'SP', '果', 'SL', '云', '层', '是', '天', '空', '的', '一', '封', 'SL', 'SP', 'AP'], text=如果云层是天空的一封信, hanzi_initials=['r', 'SP', 'g', 'SL', 'c', 'sh', 't', 'k', 'd', 'y', 'f', 'x', 'SL', 'SP', 'AP']