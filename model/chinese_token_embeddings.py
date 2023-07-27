import torch
import torch.nn as nn
from utils.chinese_to_pinyin import is_chinese
from utils.load_checkpoint import get_config, get_whisper_token_embedding
from utils.simplified_chinese_tokenizer import SimplifiedChineseTokenizer, traditional_to_simplified
from utils.word_tokenizer import WordTokenizer


class ChineseTokenEmbedding(nn.Module):
    def __init__(self, 
                 size, 
                 n_state, 
                 tokenizer: SimplifiedChineseTokenizer, 
                 model='tiny', 
                 init_embedding_from_whisper=True) -> None:
        '''
        size: embedding size of each Token
        n_state: the size of the embedding, it should be large enough for easy coding
        tokenizer: will be updated in self.update_table()
        '''
        super().__init__()
        self.chinese_token_embedding = nn.Embedding(size, n_state)
        self.tokenizer = tokenizer
        self.model = model
        self.whisper_official_token_embedding = get_whisper_token_embedding(self.model)
        self.whisper_official_word_tokenizer = WordTokenizer()
        if init_embedding_from_whisper:
            self.initialize_embedding()
        print('initial size of ChineseTokenEmbedding is', len(self.tokenizer))
        
    def initialize_embedding(self):
        '''
        1. read whisper official checkpoint, get the token embedding,
        2. map token(s) to embedding, do mean,
        3. save embedding back to self.chinese_token_embedding,
        '''
        for id in range(10):
            hanzi = self.tokenizer.get_content(id)
            tokens = self.whisper_official_word_tokenizer.encode(hanzi)[0]
            # tokens are like [123, 54], here get embedding of these tokens from token_embedding
            # and then calculate the mean of them, then save back to self.chinese_token_embedding[id]
            embeddings = self.whisper_official_token_embedding[tokens]
            mean_embedding = embeddings.mean(dim=0)
            self.chinese_token_embedding.weight.data[self.tokenizer.encode(hanzi)[0]] = mean_embedding
            
    def update_table(self, new_hanzi):
        '''
        1. when an unseen hanzi shows up, save it to the tokenizer table, and self.all_hanzi
        2. calculate the embedding of that hanzi from whisper official token embedding,
        3. save that embedding to the unused position in self.chinese_token_embedding,
        '''
        self.tokenizer.add_content(new_hanzi)
        tokens = self.whisper_official_word_tokenizer.encode(new_hanzi)[0]
        embeddings = self.whisper_official_token_embedding[tokens]
        mean_embedding = embeddings.mean(dim=0)
        self.chinese_token_embedding.weight.data[self.tokenizer.encode(new_hanzi)[0]] = mean_embedding
    
    def auto_update(self, text:str):
        '''
        1. check whether all text in tokenizer, if so, return,
        2. update_table(new_hanzi)
        '''
        for hanzi in text:
            if not self.tokenizer.contains(hanzi) \
                    and is_chinese(hanzi)\
                    and not self.tokenizer.contains(traditional_to_simplified(hanzi)):
                hanzi = traditional_to_simplified(hanzi)
                self.update_table(hanzi)
                
    def __len__(self):
        return len(self.tokenizer)
        
    def forward(self, words):
        return self.chinese_token_embedding(words)
    
    def weight(self):
        '''
        weight will be used in calculating logits, then logits will be used in loss(logits, label);
        This is why we also need to update tokenizer along with this embedding
        '''
        return self.chinese_token_embedding.weight
    
if __name__ == '__main__':
    tokenizer = SimplifiedChineseTokenizer()
    print('tokenizer size', len(tokenizer))
    cte = ChineseTokenEmbedding(5000, 384, tokenizer, 'tiny')
    print('size', len(cte))
    cte.auto_update('世界之大无奇不有2hello泗')
    print('size', len(cte))