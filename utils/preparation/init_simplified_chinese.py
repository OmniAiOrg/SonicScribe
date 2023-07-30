from data_reader.opencpop import OpenCpop
from utils.load_checkpoint import get_config
from utils.chinese_to_pinyin import is_chinese
from utils.simplified_chinese_tokenizer import traditional_to_simplified

class ChineseCharacterCollector:
    characters = set()
    def add(self, sentence: str, strict=True):
        sentence = traditional_to_simplified(sentence)
        for character in sentence:
            if is_chinese(character) or not strict:
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
                
def chinese_save_to_txt():
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
        
    ccc.add('乱捏扫楼奥寻穷窘锐森托盆虫桌揣扫俩擦暖宁嗲某农爹戳坑横嫩肯恰刷侧拴僧扽柴楼增贴忒粗润航恁坏夸帅脏床绕颏群搜噶哟盘欻顺撇掠侧耨鞥杂翁旁困嗯软汉藏草催岑谁崩这哼得诶钻弱咯佛努诺吞亏晒桑腾剖窜昂康滚顿斌狼尊拽桃闹扔孙凑卷吹挼偶谬双鸟疟摘贼扎捏囊灭乱撒')
    ccc.add('黁剋谁得佛饿丢')
    
    # save openslr33
    # save Chinese characters that's different in simplified and traditional
    # import pkgutil
    # import json
    # zhcdict = pkgutil.get_data('zhconv', 'zhcdict.json')
    # zhcdicts = json.loads(zhcdict.decode('utf-8'))
    # ccc.add(zhcdicts['SIMPONLY'].split('𠀾')[0])
    
    print(f'There are totally {len(ccc)} Chinese characters saved.')
    ccc.save_to_file(ccc_file)
    
if __name__ == '__main__':
    chinese_save_to_txt()
    