

'''
input: '<|0.74|><|0|>感<|G#4/Ab4|><|1.00|><|1.00|><|1|>受<|G#4/Ab4|><|1.42|><|1.42|><|2|>停<|F#4/Gb4|><|1.74|><|1.74|><|3|>在<|F#4/Gb4|><|2.10|><|2.10|><|4|>我<|E4|><|2.32|><|2.32|><|5|>发<|E4|><|2.84|><|2.84|><|6|>端<|D#4/Eb4|><|3.20|><|3.20|><|7|>的<|D#4/Eb4|><|3.34|><|3.34|><|8|><|SP|><|rest|><|3.44|><|3.44|><|9|>指<|E4|><|3.84|><|3.84|><|10|>尖<|E4|><|4.54|><|4.54|><|11|><|AP|><|rest|><|4.82|>'
output: {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
    }
'''
from model.whisper_official import WhisperOfficial
from utils.naive_tokenizer import WhisperTokenizer


def opencpop_to_diffsinger(tokens: list[int], tokenizer: WhisperTokenizer, template=['start','order','hanzi','note','end']) -> str:
    template_len = len(template) + 1
    assert len(tokens) % template_len == 0
    text, notes, notes_duration = [], [], []
    input_type = 'word' if 'hanzi' in template else 'other'
    for i in range(len(tokens) // template_len):
        start, end = '0.', '0.'
        note = ''
        hanzi = ''
        for j in range(template_len):
            next_1 = tokenizer.encoding.decode(tokens[i * template_len + j:i * template_len + j+1])
            next_2 = tokenizer.encoding.decode(tokens[i * template_len + j:i * template_len + j+2])
            if template[j] == 'start':
                start = next_1
            if template[j] == 'end':
                end = next_1
            if template[j] == 'hanzi':
                hanzi = next_2
            if template[j] == 'note':
                note = next_1
        text.append(hanzi)
        notes.append(note)
        notes_duration.append(float(end) - float(start))
    # chars = tokenizer.encoding.decode_tokens_bytes(tokens)
    # print(chars)
    
    return {
        'text': text,
        'notes': notes.strip(),
        'notes_duration': notes_duration.strip(),
        'input_type': input_type
    }

# input_str = '<|0.00|>还<|A4|><|0.26|><|0.26|><|1|>受<|A4|><|0.70|><|0.70|><|2|>停<|F#4/Gb4|><|1.00|><|1.00|><|2|>在<|F#4/Gb4|><|1.36|><|1.36|><|3|>我<|E4|><|1.58|><|1.58|><|4|>发<|E4|><|2.08|><|2.08|><|7|>端<|E4|><|2.40|><|2.40|><|7|>的<|E4|><|2.60|><|2.60|><|8|><|SP|><|rest|><|2.68|><|2.68|><|8|>指<|E4|><|3.10|><|3.10|><|10|>尖<|E4|><|3.78|><|3.78|><|10|><|AP|><|rest|><|4.06|><|4.06|><|11|><|SP|><|rest|><|4.06|>'

whisper_official = WhisperOfficial('tiny').to('cpu')
tokens=[50530, 17819, 121, 50397, 50581, 50581, 50403, 3416, 233, 50397, 50598, 50598, 50404, 7437, 250, 50397, 50618, 50618, 50405, 2523, 101, 50397, 50634, 50634, 50406, 1486, 239, 50392, 50646, 50646, 50407, 2129, 239, 50392, 50671, 50671, 50408, 11957, 107, 50383, 50690, 50690, 50409, 1514, 226, 50379, 50699, 50699, 50410, 50522, 50398, 50702, 50702, 50411, 8501, 229, 50392, 50722, 50722, 50412, 1530, 244, 50386, 50757, 50757, 50413, 50521, 50398, 50771]
output = opencpop_to_diffsinger(tokens, whisper_official.tokenizer)
print(output)
