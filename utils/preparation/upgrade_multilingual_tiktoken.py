import base64
import os

from utils.chinese_to_pinyin import is_chinese
from utils.simplified_chinese_tokenizer import traditional_to_simplified


def get_single_token_chinese_encoding(name: str = "multilingual"):
    vocab_path = os.path.join(os.path.dirname(__file__), "../../assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    print(n_vocab)
    single_chinese = {}
    multi_chinese = {}
    for token, rank in ranks.items():
        try:
            token = token.decode('utf-8')
        except:
            pass
        print('token', token)
        if is_chinese(token):
            single_chinese[rank] = token
        elif type(token)==str and len(token)>0:
            flag = True
            for i in range(len(token)):
                if not is_chinese(token[i]):
                    flag=False
                    continue
            if flag:
                multi_chinese[rank] = token
    return single_chinese, multi_chinese
    
def fix_single_chinese(single_chinese: dict):
    # traditional chinese to simplified chinese
    processed = set()
    for id, token in single_chinese.items():
        value = traditional_to_simplified(token)
        if value in processed:
            print(f'{value} in processed')
            value = f'<|deprecated_{id}|>'
        processed.add(value)
        single_chinese[id] = value
    return single_chinese

def fix_multi_chinese(multi_chinese: dict):
    # chinese word to nothing
    for id, token in multi_chinese.items():
        multi_chinese[id] = f'<|deprecated_{id}|>'
    return multi_chinese

def save_to_tiktoken(single_chinese:dict, multi_chinese:dict, to_name:str, from_name: str = "multilingual"):
    file = os.path.join(os.path.dirname(__file__), "../../assets", f"{to_name}.tiktoken")
    
    vocab_path = os.path.join(os.path.dirname(__file__), "../../assets", f"{from_name}.tiktoken")
    ranks = {
        int(rank): token
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    for id, token in single_chinese.items():
        ranks[id] = base64.b64encode(token.encode()).decode()
    for id, token in multi_chinese.items():
        ranks[id] = base64.b64encode(token.encode()).decode()
        
    with open(file, 'w') as f:
        for id, b64 in ranks.items():
            f.write(f'{b64} {id}\n')
        
    

if __name__ == '__main__':
    single_chinese, multi_chinese = get_single_token_chinese_encoding()
    # fix_single_chinese(single_chinese)
    fix_multi_chinese(single_chinese) # instead of traditional to simplified, remove all chinese
    fix_multi_chinese(multi_chinese)
    print('single_chinese', '|'.join(single_chinese.values()))
    print('multi_chinese', '|'.join(multi_chinese.values()))
    print(len(single_chinese), len(multi_chinese))
    save_to_tiktoken(single_chinese, multi_chinese, 'multilingual_fixed')