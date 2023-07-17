'''
A tokenizer that map Chinese characters to tokens, 1 to 1 map.
Traditional Chinese will be translate to simplifed one before map.
Tokenizer should also contains the embeddings with operators.
'''
from utils.naive_tokenizer import NaiveTokenizer
import zhconv

def traditional_to_simplified(character: str) -> str:
    return zhconv.convert(character, 'zh-cn')

class SimplifiedChineseTokenizer(NaiveTokenizer):
    def __init__(self) -> None:
        super().__init__([], '<|chinese|>')
        self.word_tokenizer = {}

    def encode(self, input:str) -> list[list[int]]:
        return [token for token in self.word_tokenizer.encoding.encode_batch([word for word in input], allowed_special="all")]

    def decode(self, input:list[list[int]]) -> str:
        return self.word_tokenizer.decode([item for sublist in input for item in sublist])
        
        
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
    from data_reader.opencpop import OpenCpop
    from utils.load_checkpoint import get_config
    from utils.chinese_to_pinyin import is_chinese
    class ChineseCharacterCollector:
        characters = set()
        def add(self, sentence: str):
            sentence = traditional_to_simplified(sentence)
            for character in sentence:
                if is_chinese(character):
                    self.characters.add(character)
                else:
                    print(f'[warn] not Chinese: {character}')
                
        def __sizeof__(self) -> int:
            return len(self.characters)
        
        def save_to_file(self, file_path:str):
            assert file_path.endswith('.txt')
            char_list = list(self.characters)
            char_list.sort()
            with open(file_path, 'w') as f:
                for i, item in enumerate(char_list):
                    f.write(f"{i} {item}\n")
                    
    ccc = ChineseCharacterCollector()
    config = get_config()
    ccc_file = config['tokenizer']['chinese_characters']
    print(f'chinese characters will be saved to {ccc_file}')
    
    # save opencpop
    oc_train = OpenCpop(train=True)
    for data in oc_train:
        ccc.add(data.original_text)
        
    oc_test = OpenCpop(train=False)
    for data in oc_test:
        ccc.add(data.original_text)
        
    # save openslr33
        
    
    # save Chinese characters that's different in simplified and traditional
    import pkgutil
    import json
    zhcdict = pkgutil.get_data('zhconv', 'zhcdict.json')
    zhcdicts = json.loads(zhcdict.decode('utf-8'))
    ccc.add(len(zhcdicts['SIMPONLY'].split('𠀾')[0]))
    
    print(f'There are totally {len(ccc)} Chinese characters saved.')
    ccc.save_to_file(ccc_file)
    

    