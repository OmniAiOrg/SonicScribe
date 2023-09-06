import jiwer
from jiwer.transforms import AbstractTransform
import re

class CerTransform(AbstractTransform):
    '''
    keep_tokens can only be special tokens, which means <|?|>
    '''
    def __init__(self, ignore_tokens:list[str]=None, keep_tokens:list[str]=None) -> None:
        self.ignore_tokens = ignore_tokens
        self.keep_tokens = keep_tokens
        assert (keep_tokens is None and ignore_tokens is not None) or \
            (keep_tokens is not None and ignore_tokens is None)
        self.pattern = r'<\|[A-Za-z0-9_\./#]+\|>'
        if keep_tokens:
            for token in keep_tokens:
                assert re.match(self.pattern, token), f'{token} not match {self.pattern}'
    
    def process_string(self, s: str):
        if self.ignore_tokens:
            for token in self.ignore_tokens:
                s = s.replace(token, '')
        elif self.keep_tokens:
            filtered_list = re.findall(self.pattern, s)
            # print(filtered_list)
            selected_list = []
            for token in filtered_list:
                if token in self.keep_tokens:
                    selected_list.append(token)
            if len(selected_list) == 0:
                selected_list.append(" ")
            return selected_list
        if len(s) == 0:
            s = " "
        return [i for i in s]
    
if __name__ == '__main__':
    cer = jiwer.wer
    ignore_tokens = ['好','!']
    ct = CerTransform(ignore_tokens)
    ct = CerTransform(keep_tokens=['<|zh1|>', '<|startofinference1|>'])
    print(cer(
        ['<|zh|><|transcribe|><|hanzi|><|startofinference|><|0.22|>展示出了如同奇偶像安东尼一样的全面犀利<|endoftext|>'], 
        ['<|zh|><|transcribe|><|hanzi|><|startofinference|>展示出了如同其偶像安东尼一样的全面犀利<|endoftext|>'], 
        ct, ct))