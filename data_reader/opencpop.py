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
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer

class OpenCpop(BaseReader):
    def __init__(self, train=True, key_filter=None) -> None:
        super().__init__(train, key_filter)
        # prepare the dataset and save to pickle for next time to use
        self.pickle_path = self.config['path']+'/temp2'
        self.NONE_TEXT = str(self.config['NONE_TEXT'])
        self.SLUR_TEXT = str(self.config['SLUR_TEXT'])
        self.AUDIO_MAX_LENGTH = int(self.config['AUDIO_MAX_LENGTH'])
        self.TEXT_MAX_LENGTH = int(self.config['TEXT_MAX_LENGTH'])
        self.SAMPLE_RATE = int(self.config['SAMPLE_RATE'])
        self.audio_transcript_pair_list = self.get_dataset(train)
        
    def get_dataset(self, train):
        dataset_txt = 'train' if train else 'test'
        data = self.parse_txt(self.path + f'/{dataset_txt}.txt')
        return self.get_audio_file_list(data, self.TEXT_MAX_LENGTH, self.AUDIO_MAX_LENGTH, self.SAMPLE_RATE, f'{dataset_txt}_audio_opencpop')
    
    def fix_line(self, id, text, phoneme, note, note_duration, phoneme_duration, slur):
        # 修复官方数据集的错误，以下是一些人工发现的错误
        # if id == '2099003690':
        #     print('slur2099003690', slur)
        if id == '2099003690' and slur == '0 0 0 0 0 0 0 0 0 0 1 0 0 0\n':
            slur = '0 0 0 0 0 0 0 1 0 0 1 0 0 0\n'
        elif id == '2003000102' and text == '如果云层是天空的一封信':
            text = '如果层是天空的一封信'
        elif id == '2034001289' and slur == '0 0 0 0 0 0 0 0 0 0 0 0 0\n':
            slur = '0 0 0 0 0 0 0 0 0 0 1 0 0\n'
        elif id == '2094003515' and text == '直到和你做了多年朋友才明白我的眼泪':
            text = '直到和你做了多年朋友才明白的眼泪'
        return id, text, phoneme, note, note_duration, phoneme_duration, slur
    
    def parse_line(self, id, text, phoneme, note, note_duration, phoneme_duration, slur):
        id, text, phoneme, note, note_duration, phoneme_duration, slur = \
            self.fix_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
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
                    assert hanzi_words_idx < len(text), f'{hanzi_words}, {hanzi_initials}'
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
        except_counter = 0
        with open(file_name, 'r') as f:
            for line in f:
                if len(line) < 10:
                    continue
                id, text, phoneme, note, note_duration, phoneme_duration, slur = line.split('|')
                try:
                    id, text, initials, finals, note, note_duration, slur, hanzi_words = self.parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
                    data.append((id, text, initials, finals, note, note_duration, slur, hanzi_words))
                except AssertionError as e:
                    except_counter += 1
                    print('parse_txt', repr(e), line)
                # text not matter, just dor debug
        if except_counter > 0:
            print(f'There are {except_counter} exceptions in total when process data')
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
            audio_relative = '/wavs/'+id+'.wav'
            audio_dir = self.path+audio_relative
            if not os.path.isfile(audio_dir):
                print(f"{audio_dir} not exist")
                continue
            audio = self.load_wave(audio_dir, sample_rate=sample_rate)[0]
            if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                print(audio_dir, len(text), len(audio))
                continue
            audio_transcript_pair_list.append((str(audio_relative), text, initials, finals, note, note_duration, slur, hanzi_words))
        # cache to pickle 
        if save_name is not None:
            if not os.path.exists(self.pickle_path):
                os.makedirs(self.pickle_path)
            with open(save_dir,'wb') as in_data:
                pickle.dump(audio_transcript_pair_list,in_data,pickle.HIGHEST_PROTOCOL)
        return audio_transcript_pair_list
        
    def get_naive_item(self, idx):
        pair = super().__getitem__(idx)
        audio_dir, text, initials, finals, note, note_duration, slur, hanzi_words = pair
        return audio_dir, hanzi_words, note_duration, text
    
    def __getitem__(self, idx) -> dict:
        pair = super().__getitem__(idx)
        audio_dir, text, initials, finals, note, note_duration, slur, hanzi_words = pair
        pinyin = [initials[i]+finals[i] if initials[i] not in ['AP', 'SP', 'SL', 'NO'] else initials[i] for i in range(len(initials))]
        note_duration_cum = [f'{sum(note_duration[:i+1]):.2f}' for i in range(len(note_duration))]
        dur_start = [f'{0/100:.2f}']+note_duration_cum[:-1]
        dur_end = note_duration_cum
        assert len(hanzi_words) == len(pinyin) and len(note) == len(dur_start) and \
            len(dur_end) == len(slur) and len(hanzi_words) == len(note) \
                and len(note) == len(dur_end)
        
        output:dict = {
            'audio': self.path + audio_dir,
            'hanzi': hanzi_words,
            'pinyin': pinyin,
            'note': note,
            'start': dur_start,
            'end': dur_end,
            'slur': slur
        }
        if self.key_filter == None:
            return output
        elif 'waveform' in self.key_filter:
            output['waveform'] = self.get_waveform(output['audio'], False)
        return {key: value for key, value in output.items() if key in self.key_filter}


if __name__ == '__main__':
    def print_data(hanzi_words, pinyin, note, dur_start, dur_end, slur):
        assert len(hanzi_words) == len(pinyin) and len(note) == len(dur_start) and \
            len(dur_end) == len(slur) and len(hanzi_words) == len(note) \
                and len(note) == len(dur_end)
        for i in range(len(hanzi_words)):
            print(f'{hanzi_words[i]}:{pinyin[i]:<6}[{note[i]:<7}] {slur[i]} ({dur_start[i]}->{dur_end[i]})')
    
    oc_train = OpenCpop()
    oc_test = OpenCpop(train=False)
    print(len(oc_test), len(oc_train))
    data = oc_test[12]
    print(data['audio'])
    import shutil
    shutil.copyfile(data['audio'], 'logs/test.wav')
    print_data(data['hanzi'], data['pinyin'], data['note'], data['start'], data['end'], data['slur'])
    '''
    能:neng  [A#3/Bb3] 0 (0.00->0.25)
    不:bu    [G4     ] 0 (0.25->0.46)
    能:neng  [G4     ] 0 (0.46->0.74)
    给:gei   [G4     ] 0 (0.74->1.00)
    我:wo    [F4     ] 0 (1.00->1.26)
    一:yi    [F4     ] 0 (1.26->1.59)
    首:shou  [D#4/Eb4] 0 (1.59->2.23)
    SP:SP    [rest   ] 0 (2.23->2.27)
    歌:ge    [D#4/Eb4] 0 (2.27->2.53)
    的:de    [F4     ] 0 (2.53->2.65)
    时:shi   [F4     ] 0 (2.65->2.88)
    SL:SL    [G4     ] 1 (2.88->3.14)
    SP:SP    [rest   ] 0 (3.14->3.25)
    间:jian  [F4     ] 0 (3.25->3.84)
    AP:AP    [rest   ] 0 (3.84->4.01)
    SP:SP    [rest   ] 0 (4.01->4.06)
    '''
    