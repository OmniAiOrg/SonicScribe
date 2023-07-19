'''
Map chinese characters in whisper tokenizer to pinyin. multiple to one.
Always choose the token with largest id.
Then check whether all pinyin in ./opencpop/cpop_pinyin2ph.txt can be mapped 
to a whisper token.
This is a prepare script that you don't need to run. The generated map can be 
found in assets folder.
For not exist pinyin, try use similar embedding to train from. like 'cui' -> 'cu'
'''

from utils.word_tokenizer import WordTokenizer
from pypinyin import pinyin, lazy_pinyin, Style
from utils.opencpop.map import cpop_pinyin2ph_func
from utils.simplified_chinese_tokenizer import traditional_to_simplified

def is_chinese(c):
    # return True
    if not ('\u4e00' <= c <= '\u9fa5'):
        return False
    return True

def check_chinese_words_with_single_whisper_token():
    word_tokenizer = WordTokenizer()
    map = word_tokenizer.word_tokenizer.encoding._mergeable_ranks
    map_size = len(map)
    counter = 0
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
                print(counter, loc, word)
            
def get_all_chinese_whisper_token():
    pinyin_keys = set(cpop_pinyin2ph_func().keys())
    word_tokenizer = WordTokenizer()
    map = word_tokenizer.word_tokenizer.encoding._mergeable_ranks
    map_size = len(map)
    test_num = 100000
    pinyin_map = {}
    for i in range(map_size):
        loc = map_size - i - 1
        word = word_tokenizer.decode([[loc]])
        if len(word) == 1 and is_chinese(word):
            py = lazy_pinyin(word)[0]
            pinyin_map[py] = loc
            print(loc, word, py)
        if test_num < 0:
            break
        test_num -= 1
    print(len(pinyin_map))
    print(pinyin_keys-set(pinyin_map.keys()))

if __name__ == '__main__':
    check_chinese_words_with_single_whisper_token()