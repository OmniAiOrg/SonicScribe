'''
A tokenizer that map Chinese characters to tokens, 1 to 1 map.
Traditional Chinese will be translate to simplifed one before map.
Tokenizer should also contains the embeddings with operators.
'''
from utils.naive_tokenizer import NaiveTokenizer
import zhconv
from utils.load_checkpoint import get_config, get_whisper_token_embedding

def traditional_to_simplified(character: str) -> str:
    return zhconv.convert(character, 'zh-cn')


class SimplifiedChineseTokenizer(NaiveTokenizer):
    def __init__(self, task:str='<|chinese|>') -> None:
        super().__init__([], task)
        config = get_config()
        self.chinese_map_file = config['tokenizer']['simplified_chinese_characters']
        print(f'load chinese characters from {self.chinese_map_file}')
        self.all_hanzi = FileUpdater(self.chinese_map_file)
        for i in range(len(self.all_hanzi)):
            idx = len(self.vocabs)
            self.vocabs.append(self.get_content(i))
            self.converter[self.get_content(i)] = idx
    
    def add_content(self, new_hanzi):
        self.all_hanzi.add_content(new_hanzi)
        idx = len(self.vocabs)
        self.vocabs.append(new_hanzi)
        self.converter[new_hanzi] = idx
        
    def get_content(self, id):
        return self.all_hanzi.get_content(id)
        
    def contains(self, hanzi):
        return self.all_hanzi.check_content_exist(hanzi)
    
        

class FileUpdater:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content_to_id = {}
        self.id_to_content = {}
        with open(file_path, 'r') as f:
            for line in f:
                if len(line) > 2:
                    content, id = line.strip().split()
                    self.content_to_id[content] = int(id)
                    self.id_to_content[int(id)] = content

    def get_id(self, content):
        return self.content_to_id.get(content)

    def get_content(self, id):
        return self.id_to_content.get(id)

    def check_content_exist(self, content):
        return content in self.content_to_id

    def add_content(self, content):
        assert len(content) == 1, f'content={content}, size != 1'
        if not self.check_content_exist(content):
            new_id = len(self.id_to_content)
            self.id_to_content[new_id] = content
            self.content_to_id[content] = new_id
            with open(self.file_path, 'a') as f:
                f.write(f"{content} {new_id}\n")

    def __len__(self) -> int:
        return len(self.id_to_content)
    
if __name__ == '__main__':
    '''
    To use the pretrained model parameters, we may first transfer previous tokens
    to simgle character token.
    1. Save all simplified chinese in a map with id start from 0;
    2. Save the embedding into a single file which is easy to read and use;
    3. Go through all chineses characters, try concat/add... embeddings into single one;
    4. Save these embeddings into single file with operators;
    5. The the embeddings and tokenizer can be load for training; 
    '''
    from utils.preparation.init_simplified_chinese import chinese_save_to_txt
    chinese_save_to_txt()
    
    word_tokenizer = SimplifiedChineseTokenizer()
    print('before add_content, tokenizer size', len(word_tokenizer))
    
    cc = word_tokenizer.encode(["语", "文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode(cc)
    print(f'decode batch = {cc}')
    
    word_tokenizer.add_content("文")
    print('after add_content, tokenizer size', len(word_tokenizer))
    cc = word_tokenizer.encode(["语", "文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode(cc)
    print(f'decode batch = {cc}')
    
    cc = word_tokenizer.encode(["<|startoftranscript|>", "语", "如", "語", "文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode(cc)
    print(f'decode batch = {cc}')
    cc = word_tokenizer.encode(["晒", "鹅", "剋", "文"])
    print(f'encode batch = {cc}')
    cc = word_tokenizer.decode(cc)
    print(f'decode batch = {cc}')
    
    
    
    