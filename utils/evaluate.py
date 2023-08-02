import jiwer
from jiwer.transforms import AbstractTransform

class CerTransform(AbstractTransform):
    def __init__(self, ignore_tokens:list[str]) -> None:
        self.ignore_tokens = ignore_tokens
    
    def process_string(self, s: str):
        for token in self.ignore_tokens:
            s = s.replace(token, '')
        return [i for i in s]
    
if __name__ == '__main__':
    cer = jiwer.cer
    ignore_tokens = ['好','!']
    ct = CerTransform(ignore_tokens)
    print(cer(['你真是好棒'], ['你真是棒!'], ct, ct))