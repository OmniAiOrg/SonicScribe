from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from utils.naive_tokenizer import NaiveTokenizer

'''
In this tokenizer, all Chinese characters will be convert to simplified Chinese,
and then check if it can be encode to single token, if not (like "语"), 
1. first convert to traditional Chinese ("語"), check again, if still not,
2. add it to special_tokens, copy the weight of all its tokens, do mean, set, finetune.

There will be a local Map file saved in assets folder, and when the model is trained, 
the assets files will be used for inference.
On the other hand, this would also work for other languages, like English.
For English, we may first convert word to phoneme, and then save phoneme to special_tokens.

Why do this? For singings, notes match phoneme one by one. so each token match one phoneme
makes things easy.

TODO: Is there any better methods that no need to train word embedding? like padding on 
notes and timestamp?
'''
class WhisperOfficialTokenizer(NaiveTokenizer):
    def __init__(self) -> None:
        super().__init__([], '<|word|>')
        self.word_tokenizer = get_tokenizer(True, language='zh', task='transcribe')
        
    def encode_list(self, input:list[str]) -> list[list[int]]:
        return [token for token in self.word_tokenizer.encoding.encode_batch(input, allowed_special="all")]
    
    def encode(self, input:str) -> list[list[int]]:
        return [token for token in self.word_tokenizer.encoding.encode_batch([word for word in input], allowed_special="all")]
    
    def decode_list(self, input:list[list[int]]) -> list[str]:
        return [self.word_tokenizer.decode(token) for token in input]
    
    def decode(self, input:list[list[int]]) -> str:
        return self.word_tokenizer.decode([item for sublist in input for item in sublist])
        
# class WordEmbeddingInitializer():
#     def __init__(self) -> None:
#         self.word_tokenizer = WordTokenizer()
        
#     def 
        
        
        
if __name__ == '__main__':
    word_tokenizer = WhisperOfficialTokenizer()
    cc = word_tokenizer.encode("语文")
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode(cc)
    print(f'decode batch = {cc}')
    cc = word_tokenizer.encode_list(["<|startoftranscript|>", "语", "如果", "語", "語文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode_list(cc)
    print(f'decode batch = {cc}')
    print(f'sot = {word_tokenizer.word_tokenizer.sot}')
    cc = word_tokenizer.encode_list(["晒", "鹅", "剋", "語文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode_list(cc)
    print(f'decode batch = {cc}')