'''
Unlike tiktoken, this naive tokenizer is a 1 to 1 tokenizer, which will not
map to multiple tokens.
'''

import base64
import os
from typing import Dict, List, Optional, Tuple, Union
from functools import cached_property, lru_cache
import tiktoken

from whisper.tokenizer import Tokenizer, LANGUAGES, TO_LANGUAGE_CODE
from utils.load_checkpoint import get_config
from utils.note import notes
from utils.ph import phs

class NaiveTokenizer:
    
    def __init__(self, vocabs:List[str], task:str) -> None:
        self.tasks = ['<|initial|>','<|final|>','<|note|>','<|slur|>','<|word|>', '<|note_duration|>', '<|chinese|>', '<|pinyin|>', '<|tone|>']
        assert task in self.tasks
        self.spetial_code:List[str] = ['<|startoftranscript|>','<|transcribe|>','<|notimestamps|>','<|endoftext|>','<|pad|>','<|pad_label|>','<|unknow|>']
        self.vocabs:List[str] = [*self.spetial_code, task, *vocabs]
        self.converter: Dict[str, int] = {}
        
        for i, vocab in enumerate(self.vocabs):
            self.converter[vocab] = i

        self.sot: int = self.converter["<|startoftranscript|>"]
        self.transcribe: int = self.converter["<|transcribe|>"]
        self.no_timestamps: int = self.converter["<|notimestamps|>"]
        self.eot: int = self.converter["<|endoftext|>"]
        self.task: int = self.converter[task]
        self.pad: int = self.converter["<|pad|>"]
        self.pad_label: int = self.converter["<|pad_label|>"]
        self.unknow: int = self.converter["<|unknow|>"]
        
    def __len__(self):
        return len(self.vocabs)

    def encode(self, text: List[str], strict = False, default=-1, **kwargs) -> List[int]:
        if strict:
            token_ids = [self.converter[t] for t in text]
        else:
            token_ids = [self.converter[t] if t in self.converter else default for t in text]
        return token_ids

    def decode(self, token_ids: List[int], stop_at:int=-100, compact=False, **kwargs) -> Union[str, list[str]]:
        if compact:
            strs = ''
            for c in token_ids:
                if c == stop_at:
                    break
                if c in self.sot_task_so_on or c in [self.pad, self.pad_label, self.eot]:
                    continue
                strs += self.vocabs[c] if c is not None and c < len(self.vocabs) else "☐"
                strs += ' '
            return strs # str
        else:
            output = []
            for c in token_ids:
                if c == stop_at:
                    break
                output.append(self.vocabs[c] if c is not None and c < len(self.vocabs) else "☐")
            return output # list[str]
    
    @cached_property
    def sot_task_so_on(self) -> Tuple[int]:
        return tuple([self.sot, self.task])

class DummyTokenizer(NaiveTokenizer):
    def encode(self, text: List[str], **kwargs) -> List[int]:
        token_ids = [0 for t in text]
        return token_ids

    def decode(self, token_ids: List[int], stop_at=-100, **kwargs) -> str:
        return 'dummy'
    
class WhisperTokenizer(Tokenizer):
    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]
        self.note: int = self.special_tokens["<|note|>"]
        self.hanzi: int = self.special_tokens["<|hanzi|>"]
        self.pinyin: int = self.special_tokens["<|pinyin|>"]

        langs = tuple(LANGUAGES.keys())
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)
        
    @cached_property
    def pad(self) -> int:
        return self.special_tokens["<|pad|>"]
    
    @cached_property
    def soi(self) -> int:
        return self.special_tokens["<|startofinference|>"]
    
    @cached_property
    def notes_begin(self) -> int:
        return self.special_tokens["<|A#3/Bb3|>"]
    
    @cached_property
    def order(self) -> int:
        return self.special_tokens["<|order|>"]
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin or t >= self.notes_begin]
        if 'stop_at' in kwargs:
            stop_at:int = kwargs['stop_at']
            del kwargs['stop_at']
            target_index = token_ids.index(stop_at)+1 if stop_at in token_ids else len(token_ids)+1
            token_ids = token_ids[:target_index]
        return self.encoding.decode(token_ids, **kwargs)
        
@lru_cache(maxsize=None)
def get_encoding(name: str = "multilingual"):
    vocab_path = os.path.join(os.path.dirname(__file__), "../assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}
    max_text_size = get_config('data_reader/dataset_config.yaml')['BaseReader']['TEXT_MAX_LENGTH']
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        # *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        *[f'<|{i}|>' for i in notes],
        *[f'<|{i}|>' for i in range(max_text_size)],
        "<|SL|>",
        "<|pad|>",
        "<|AP|>",
        "<|SP|>",
        "<|note|>",
        "<|hanzi|>",
        "<|pinyin|>",
        "<|startofinference|>",
        "<|order|>",
        # For new special tokens, add them before this line
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> WhisperTokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual_fixed"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name)

    return WhisperTokenizer(encoding=encoding, language=language, task=task)

    
if __name__ == '__main__':
    slur_tokenizer = NaiveTokenizer(['0', '1'], '<|slur|>')
    en = [*slur_tokenizer.sot_task_so_on] + slur_tokenizer.encode(['0', '1'])
    print(en)
    de = slur_tokenizer.decode(en)
    print(de)
    print('-------')
    
    whisper_tokenizer = get_tokenizer(multilingual=True, language='zh', task='transcibe')
    print(whisper_tokenizer.encoding.n_vocab)
    en = whisper_tokenizer.encode('<|1|>世<|A3|><|2|>界<|C#5/Db5|>', allowed_special="all")
    print(en)
    de = whisper_tokenizer.decode(en)
    print(de)
    print('-------')
    
    en = whisper_tokenizer.encode('<|1|>sh<|A3|><|2|>i<|C#5/Db5|>', allowed_special="all")
    print(en)
    de = whisper_tokenizer.decode(en)
    print(de)
    print('-------')
    
    print(whisper_tokenizer.encode('<|SL|>', allowed_special="all"))
    print(whisper_tokenizer.encoding.encode_single_token('<|SL|>'))