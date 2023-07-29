import torch.nn as nn
from utils.chinese_to_pinyin import get_all_chinese_with_multuple_whisper_token, get_all_chinese_with_single_whisper_token, is_chinese
from utils.load_checkpoint import get_config, get_whisper_token_embedding
from utils.naive_tokenizer import NaiveTokenizer
from utils.tokenizers import pinyin_tokenizer


class PinyinTokenEmbedding(nn.Module):
    def __init__(self, 
                 n_state, 
                 tokenizer: NaiveTokenizer, 
                 model='tiny', 
                 init_embedding_from_whisper=True) -> None:
        '''
        size: embedding size of each Token
        n_state: the size of the embedding, it should be large enough for easy coding
        tokenizer: will be updated in self.update_table()
        '''
        super().__init__()
        self.pinyin_token_embedding = nn.Embedding(len(tokenizer), n_state)
        self.tokenizer = tokenizer
        self.model = model
        if init_embedding_from_whisper:
            self.initialize_embedding()
        print('initial size of PinyinTokenEmbedding is', len(self.tokenizer))
        
    def initialize_embedding(self):
        '''
        1. read whisper official checkpoint, get the token embedding,
        2. map token(s) to embedding, do mean,
        3. save embedding back to self.chinese_token_embedding,
        '''
        single_token_map = get_all_chinese_with_single_whisper_token(False)
        multiple_token_map = get_all_chinese_with_multuple_whisper_token(False)
        whisper_official_token_embedding = get_whisper_token_embedding(self.model)
        for id in range(len(tokenizer)):
            pinyin = self.tokenizer.decode([id])[0]
            if pinyin in single_token_map:
                tokens = [single_token_map[pinyin]]
            elif pinyin in multiple_token_map:
                tokens = multiple_token_map[pinyin]
            else:
                continue
            # tokens are like [123, 54], here get embedding of these tokens from token_embedding
            # and then calculate the mean of them, then save back to self.pinyin_token_embedding[id]
            embeddings = whisper_official_token_embedding[tokens]
            mean_embedding = embeddings.mean(dim=0)
            self.pinyin_token_embedding.weight.data[self.tokenizer.encode(pinyin)[0]] = mean_embedding
            
    def __len__(self):
        return len(self.tokenizer)
        
    def forward(self, words):
        return self.pinyin_token_embedding(words)
    
    def weight(self):
        '''
        weight will be used in calculating logits, then logits will be used in loss(logits, label);
        This is why we also need to update tokenizer along with this embedding
        '''
        return self.pinyin_token_embedding.weight
    
if __name__ == '__main__':
    tokenizer = pinyin_tokenizer
    print('tokenizer size', len(tokenizer))
    cte = PinyinTokenEmbedding(384, tokenizer, 'tiny')
    print('size', len(cte))