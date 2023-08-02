'''
Unlike tiktoken, this naive tokenizer is a 1 to 1 tokenizer, which will not
map to multiple tokens.
'''

from typing import Dict, List, Optional, Tuple, Union
from functools import cached_property

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
                if c in self.sot_task_so_on or c in [self.pad, self.pad_label]:
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
    
if __name__ == '__main__':
    slur_tokenizer = NaiveTokenizer(['0', '1'], '<|slur|>')
    en = [*slur_tokenizer.sot_task_so_on] + slur_tokenizer.encode(['0', '1'])
    print(en)
    de = slur_tokenizer.decode(en)
    print(de)