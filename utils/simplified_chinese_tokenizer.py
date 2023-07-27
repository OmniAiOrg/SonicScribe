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
    def __init__(self) -> None:
        super().__init__([], '<|chinese|>')
        config = get_config()
        self.chinese_map_file = config['tokenizer']['simplified_chinese_characters']
        print(f'load chinese characters from {self.chinese_map_file}')
        self.all_hanzi = FileUpdater(self.chinese_map_file)
    
    def add_content(self, new_hanzi):
        self.all_hanzi.add_content(new_hanzi)
        
    def get_content(self, id):
        return self.all_hanzi.get_content(id)
        
    def contains(self, hanzi):
        return self.all_hanzi.check_content_exist(hanzi)
    
    def __len__(self) -> int:
        return len(self.all_hanzi)

    def encode(self, input:str) -> list[int]:
        return [self.all_hanzi.get_id(i) for i in input]

    def decode(self, input:list[int]) -> str:
        return [self.all_hanzi.get_content(i) for i in input]
        

class FileUpdater:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content_to_id = {}
        self.id_to_content = {}
        with open(file_path, 'r') as f:
            for line in f:
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
    from data_reader.opencpop import OpenCpop
    from utils.load_checkpoint import get_config
    from utils.chinese_to_pinyin import is_chinese
    from utils.word_tokenizer import WordTokenizer
    
    class ChineseCharacterCollector:
        characters = set()
        def add(self, sentence: str):
            sentence = traditional_to_simplified(sentence)
            for character in sentence:
                if is_chinese(character):
                    self.characters.add(character)
                else:
                    print(f'[warn] not Chinese: {character}')
                
        def __len__(self) -> int:
            return len(self.characters)
        
        def save_to_file(self, file_path:str):
            assert file_path.endswith('.txt')
            char_list = list(self.characters)
            char_list.sort()
            with open(file_path, 'w') as f:
                for i, item in enumerate(char_list):
                    f.write(f"{item} {i}\n")
                    
    def chinese_save_depracated():
        ccc = ChineseCharacterCollector()
        config = get_config()
        ccc_file = config['tokenizer']['simplified_chinese_characters']
        print(f'chinese characters will be saved to {ccc_file}')
        
        # save opencpop
        oc_train = OpenCpop(train=True)
        for i in range(len(oc_train)):
            audio_dir, hanzi_words, note_duration, text = oc_train.get_naive_item(i)
            ccc.add(text)
            
        oc_test = OpenCpop(train=False)
        for i in range(len(oc_test)):
            audio_dir, hanzi_words, note_duration, text = oc_test.get_naive_item(i)
            ccc.add(text)
            
        # save openslr33
        # save Chinese characters that's different in simplified and traditional
        import pkgutil
        import json
        zhcdict = pkgutil.get_data('zhconv', 'zhcdict.json')
        zhcdicts = json.loads(zhcdict.decode('utf-8'))
        # ccc.add(zhcdicts['SIMPONLY'].split('𠀾')[0])
        
        print(f'There are totally {len(ccc)} Chinese characters saved.')
        ccc.save_to_file(ccc_file)
        
    chinese_save_depracated()
        