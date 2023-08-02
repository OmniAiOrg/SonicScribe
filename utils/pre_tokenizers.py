from typing import List
from utils.naive_tokenizer import NaiveTokenizer, DummyTokenizer
from whisper.tokenizer import get_tokenizer
from utils.note import notes
from utils.opencpop.map import cpop_pinyin2ph_func
from utils.ph import get_initials_and_finals
from utils.load_checkpoint import get_config
from utils.simplified_chinese_tokenizer import SimplifiedChineseTokenizer

config = get_config('data_reader/dataset_config.yaml')
NONE_TEXT = str(config['BaseReader']['NONE_TEXT'])
SLUR_TEXT = str(config['BaseReader']['SLUR_TEXT'])
MAX_SECS = int(config['BaseReader']['MAX_SECS'])
AP_SP_SL = ['AP','SP','SL','NO']
_initials, _finals = get_initials_and_finals()

class PinyinTokenizer(NaiveTokenizer):
    def __init__(self, task: str) -> None:
        self.pinyin_map = cpop_pinyin2ph_func()
        for k, v in self.pinyin_map.items():
            self.pinyin_map[k] = v.replace(" ", "")
        vocabs: List[str] = AP_SP_SL+sorted(list(self.pinyin_map.values()))
        super().__init__(vocabs, task)
        
    def encode(self, text: List[str], strict=False, **kwargs) -> List[int]:
        text = [i if i not in self.pinyin_map.keys() else self.pinyin_map[i] for i in text]
        return super().encode(text, strict, **kwargs)

initials_tokenizer = NaiveTokenizer(AP_SP_SL+[NONE_TEXT, SLUR_TEXT]+_initials, '<|initial|>')
finals_tokenizer = NaiveTokenizer(AP_SP_SL+_finals, '<|final|>')
note_tokenizer = NaiveTokenizer(notes, '<|note|>')
slur_tokenizer = NaiveTokenizer(['0', '1'], '<|slur|>')
duration_tokenizer = NaiveTokenizer([f'{t/100:.2f}' for t in range(MAX_SECS * 100)], '<|note_duration|>')
word_tokenizer = SimplifiedChineseTokenizer(AP_SP_SL, task='<|chinese|>')
pinyin_tokenizer = PinyinTokenizer('<|pinyin|>')
tone_tokenizer = NaiveTokenizer(['0', '1', '2', '3', '4', '5'], '<|tone|>')

all_tokenizers:dict[str, NaiveTokenizer] = {
    'initials': initials_tokenizer,
    'finals': finals_tokenizer,
    'note': note_tokenizer,
    'slur': slur_tokenizer,
    'duration': duration_tokenizer,
    'start': duration_tokenizer,
    'end': duration_tokenizer,
    'hanzi': word_tokenizer,
    'pinyin': pinyin_tokenizer,
    'tone': tone_tokenizer
}

if __name__ == '__main__':
    for tokenizer in all_tokenizers.values():
        assert tokenizer.pad == word_tokenizer.pad
        assert tokenizer.pad_label == word_tokenizer.pad_label
        assert tokenizer.unknow == word_tokenizer.unknow
    
    print(pinyin_tokenizer.encode(['ni','lai','le']))
    print(pinyin_tokenizer.decode([236, 180, 184]))
    
    print(word_tokenizer.encode(['我','爱','你']))
    print(word_tokenizer.decode([1336, 2259, 174]))
    
    print(duration_tokenizer.encode(['0.00', '1.28', '4.35']))
    print(duration_tokenizer.decode([8, 136, 443]))
    
    print(slur_tokenizer.encode(['0','1','1']))
    print(slur_tokenizer.decode([8, 9, 9]))
    
    print(note_tokenizer.encode(['<|endoftext|>', 'A#4/Bb4', 'A5']))
    print(note_tokenizer.decode([3, 9, 12]))
    
    print(finals_tokenizer.encode(['ei', 'en', 'eng']))
    print(finals_tokenizer.decode([18, 19, 20]))
    
    print(initials_tokenizer.encode(['l','m','n']))
    print(initials_tokenizer.decode([23, 24, 25]))
    
    print(tone_tokenizer.encode(['0', '1', '2']))
    print(tone_tokenizer.decode([8, 9, 10]))
    