from utils.naive_tokenizer import NaiveTokenizer, DummyTokenizer
from whisper.tokenizer import get_tokenizer
from utils.note import notes
from utils.ph import get_initials_and_finals
from utils.load_checkpoint import get_config

config = get_config('data_reader/dataset_config.yaml')
NONE_TEXT = str(config['BaseReader']['NONE_TEXT'])
SLUR_TEXT = str(config['BaseReader']['SLUR_TEXT'])
MAX_SECS = int(config['BaseReader']['MAX_SECS'])
_initials, _finals = get_initials_and_finals()
initials_tokenizer = NaiveTokenizer(_initials+[NONE_TEXT, SLUR_TEXT]+['AP','SP'], '<|initial|>')
finals_tokenizer = NaiveTokenizer(_finals+['AP','SP'], '<|final|>')
note_tokenizer = NaiveTokenizer(notes, '<|note|>')
slur_tokenizer = NaiveTokenizer(['0', '1', '2'], '<|slur|>')
note_duration_tokenizer = NaiveTokenizer([f'<|{t/100:.2f}|>' for t in range(MAX_SECS * 100)], '<|note_duration|>')
# whisper_zh_tokenizer = get_tokenizer(True, language='zh', task='transcribe')

words_tokenizer = DummyTokenizer([], '<|word|>')