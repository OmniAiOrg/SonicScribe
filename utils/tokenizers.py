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
AP_SP = ['AP','SP']
_initials, _finals = get_initials_and_finals()
initials_tokenizer = NaiveTokenizer(_initials+[NONE_TEXT, SLUR_TEXT]+AP_SP, '<|initial|>')
finals_tokenizer = NaiveTokenizer(_finals+AP_SP, '<|final|>')
note_tokenizer = NaiveTokenizer(notes, '<|note|>')
slur_tokenizer = NaiveTokenizer(['0', '1', '2'], '<|slur|>')
duration_tokenizer = NaiveTokenizer([f'{t/100:.2f}' for t in range(MAX_SECS * 100)], '<|note_duration|>')
word_tokenizer = SimplifiedChineseTokenizer(task='<|chinese|>')
pinyin_tokenizer = NaiveTokenizer(sorted(list(cpop_pinyin2ph_func().keys()))+AP_SP, '<|pinyin|>')
tone_tokenizer = NaiveTokenizer(['0', '1', '2', '3', '4', '5'], '<|tone|>')

if __name__ == '__main__':
    print(pinyin_tokenizer.encode(['ni','lai','le']))
    print(pinyin_tokenizer.decode([229, 173, 177]))
    
    print(word_tokenizer.encode(['我','爱','你']))
    print(word_tokenizer.decode([662, 1060, 103]))
    
    print(duration_tokenizer.encode(['0.00', '1.28', '4.35']))
    print(duration_tokenizer.decode([123, 133, 440]))
    
    print(slur_tokenizer.encode(['0','1','2']))
    print(slur_tokenizer.decode([5, 6, 7]))
    
    print(note_tokenizer.encode(['<|endoftext|>', 'A#4/Bb4', 'A5']))
    print(note_tokenizer.decode([3, 6, 9]))
    
    print(finals_tokenizer.encode(['ei', 'en', 'eng']))
    print(finals_tokenizer.decode([11, 12, 13]))
    
    print(initials_tokenizer.encode(['l','m','n']))
    print(initials_tokenizer.decode([14, 15, 16]))
    