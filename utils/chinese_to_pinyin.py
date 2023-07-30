'''
Map chinese characters in whisper tokenizer to pinyin. multiple to one.
Always choose the token with largest id.
Then check whether all pinyin in ./opencpop/cpop_pinyin2ph.txt can be mapped 
to a whisper token.
This is a prepare script that you don't need to run. The generated map can be 
found in assets folder.
For not exist pinyin, try use similar embedding to train from. like 'cui' -> 'cu'
'''

from utils.load_checkpoint import get_config
from utils.word_tokenizer import WhisperOfficialTokenizer
from pypinyin import pinyin, lazy_pinyin, Style
from utils.opencpop.map import cpop_pinyin2ph_func
from utils.simplified_chinese_tokenizer import FileUpdater, traditional_to_simplified

cjk_ranges = [
        ( 0x4E00,  0x62FF),
        ( 0x6300,  0x77FF),
        ( 0x7800,  0x8CFF),
        ( 0x8D00,  0x9FCC),
        ( 0x3400,  0x4DB5),
        (0x20000, 0x215FF),
        (0x21600, 0x230FF),
        (0x23100, 0x245FF),
        (0x24600, 0x260FF),
        (0x26100, 0x275FF),
        (0x27600, 0x290FF),
        (0x29100, 0x2A6DF),
        (0x2A700, 0x2B734),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F)
    ]

def is_cjk(char):
    '''
    this function was copied from https://stackoverflow.com/a/52837006
    '''
    char = ord(char)
    for bottom, top in cjk_ranges:
        if char >= bottom and char <= top:
            return True
    return False

def is_chinese(c):
    # return True
    if len(c) != 1:
        return False
    return is_cjk(c)

def check_chinese_words_with_single_whisper_token(verbose=True):
    '''
    return a list of chinese_words_with_single_whisper_token
    eg:
    251 10440 什么
    252 8861 一下
    253 8034 真的
    254 7758 知道
    '''
    word_tokenizer = WhisperOfficialTokenizer()
    map = word_tokenizer.word_tokenizer.encoding._mergeable_ranks
    map_size = len(map)
    counter = 0
    chinese_words_with_single_whisper_token = []
    for i in range(map_size):
        loc = map_size - i - 1
        word = word_tokenizer.decode([[loc]])
        if len(word) > 1:
            all_chinese = True
            for w in word:
                if not is_chinese(w):
                    all_chinese = False
            if not all_chinese:
                continue
            simplified_word = traditional_to_simplified(word)
            if word == simplified_word:
                counter += 1
                if verbose:
                    print(counter, loc, word)
                chinese_words_with_single_whisper_token.append((counter, loc, word))
    return chinese_words_with_single_whisper_token
            
def get_all_chinese_with_single_whisper_token(verbose=True):
    '''
    eg:
    575 走 zou
    9574 之 zhi
    9572 其 qi
    9497 们 men
    9487 次 ci
    9463 覺 jue
    '''
    pinyin_keys = set(cpop_pinyin2ph_func().keys())
    word_tokenizer = WhisperOfficialTokenizer()
    map = word_tokenizer.word_tokenizer.encoding._mergeable_ranks
    map_size = len(map)
    pinyin_map = {}
    for i in range(map_size):
        loc = map_size - i - 1
        word = word_tokenizer.decode([[loc]])
        if len(word) == 1 and is_chinese(word):
            py = lazy_pinyin(word)[0]
            pinyin_map[py] = i
            if verbose:
                print(loc, word, py)
    if verbose:
        print('should have',len(pinyin_keys),'exist',len(pinyin_map))
        print('lost', pinyin_keys-set(pinyin_map.keys()))
    return pinyin_map

def get_all_chinese_with_multuple_whisper_token(verbose=True):
    single_pinyin_map = get_all_chinese_with_single_whisper_token(verbose)
    need_pinyin_keys = set(cpop_pinyin2ph_func().keys()) - set(single_pinyin_map.keys())
    word_tokenizer = WhisperOfficialTokenizer()
    
    config = get_config()
    chinese_map_file = config['tokenizer']['simplified_chinese_characters']
    if verbose:
        print(f'load chinese characters from {chinese_map_file}')
    all_hanzi = FileUpdater(chinese_map_file)
    pinyin_map = {}
    for i in range(len(all_hanzi)):
        hanzi = all_hanzi.get_content(i)
        py = lazy_pinyin(hanzi)[0]
        tokens = word_tokenizer.encode(hanzi)[0]
        if py not in need_pinyin_keys:
            continue
        pinyin_map[py] = tokens
        need_pinyin_keys -= set([py])
        if verbose:
            print(tokens, hanzi, py)
    if verbose:
        print('lost', sorted(list(need_pinyin_keys)))
    return pinyin_map
        
            


if __name__ == '__main__':
    # check_chinese_words_with_single_whisper_token()
    single_token_map = get_all_chinese_with_single_whisper_token(False)
    multiple_token_map = get_all_chinese_with_multuple_whisper_token(False)