'''
Unlike tiktoken, this naive tokenizer is a 1 to 1 tokenizer, which will not
map to multiple tokens.
'''

from typing import Dict, List, Optional, Tuple
from functools import cached_property

class NaiveTokenizer:
    
    def __init__(self, vocabs:List[str], task:str) -> None:
        assert task in ['<|initial|>','<|final|>','<|note|>','<|slur|>','<|word|>', '<|note_duration|>'], task
        self.spetial_code:List[str] = ['<|startoftranscript|>','<|transcribe|>','<|notimestamps|>','<|endoftext|>']
        self.vocabs:List[str] = [*self.spetial_code, task, *vocabs]
        self.converter: Dict[str, int] = {}
        self.size = len(self.vocabs)
        
        for i, vocab in enumerate(self.vocabs):
            self.converter[vocab] = i

        self.sot: int = self.converter["<|startoftranscript|>"]
        self.transcribe: int = self.converter["<|transcribe|>"]
        self.no_timestamps: int = self.converter["<|notimestamps|>"]
        self.eot: int = self.converter["<|endoftext|>"]
        self.task: int = self.converter[task]

    def encode(self, text: List[str], **kwargs) -> List[int]:
        token_ids = [self.converter[t] for t in text]
        return token_ids

    def decode(self, token_ids: List[int], stop_at=-100, **kwargs) -> str:
        strs = ''
        for c in token_ids:
            if c == stop_at:
                break
            strs += self.vocabs[c]
            strs += ' '
        return strs
    
    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple([self.sot, self.task, self.transcribe] + [self.no_timestamps])

class DummyTokenizer(NaiveTokenizer):
    def encode(self, text: List[str], **kwargs) -> List[int]:
        token_ids = [0 for t in text]
        return token_ids

    def decode(self, token_ids: List[int], stop_at=-100, **kwargs) -> str:
        return 'dummy'