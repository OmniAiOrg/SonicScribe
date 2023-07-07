from ph import get_initials_and_finals

def parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur):
    none_text = 'NO'
    slur_text = 'SL'
    # 在此修改格式，使满足元辅音区分后共同预测，简化预测过程
    initials, finals = get_initials_and_finals()
    phoneme = phoneme.split(' ')
    note = note.split(' ')
    note_duration = note_duration.split(' ')
    slur = slur.split('\n')[0].split(' ')
    hanzi_initials = []
    hanzi_finals = []
    hanzi_note = []
    hanzi_note_duration = []
    hanzi_slur = []
    for i in range(len(phoneme)):
        p = phoneme[i]
        assert slur[i] in ['0', '1']
        if p not in initials and p not in finals:
            # p in ['AP', 'SP']
            hanzi_initials.append(p)
            hanzi_finals.append(p)
        elif p in initials:
            # 元音就一定有辅音，然后跳过等到辅音存储
            assert i != len(phoneme)-1 and phoneme[i+1] in finals
            continue
        elif p in finals:
            # 辅音考虑以下情况
            # 1. 前一个是元音的辅音，不slur，这就是一般汉字“sh ui”
            # 2. 前一个是元音的辅音，slur，不应该存在
            # 3. 前一个是辅音的辅音，不slur，比如“好啊”
            # 4. 前一个是辅音的辅音，slur，也就是延音
            # 5. 前一个都不是的辅音，不slur，“AP 啊”
            # 6. 前一个都不是的辅音，slur，不应该存在
            if phoneme[i-1] in initials:
                assert slur[i] == '0', 'slur[i] can only be 0'
                assert note[i] == note[i-1]
                assert note_duration[i] == note_duration[i-1]
                hanzi_initials.append(phoneme[i-1])
                hanzi_finals.append(p)
            elif phoneme[i-1] in finals:
                if slur[i] == '0':
                    hanzi_initials.append(none_text)
                    hanzi_finals.append(p)
                elif slur[i] == '1':
                    hanzi_initials.append(slur_text)
                    hanzi_finals.append(p)
            else:
                assert phoneme[i-1] in ['AP', 'SP'], f'phoneme[i-1]={phoneme[i-1]} not in AP/SP'
                assert slur[i] == '0', 'slur[i] can only be 0'
                hanzi_initials.append(none_text)
                hanzi_finals.append(p)
        hanzi_note.append(note[i])
        hanzi_note_duration.append(note_duration[i])
        hanzi_slur.append(slur[i])
    return id, text, hanzi_initials, hanzi_finals, hanzi_note, hanzi_note_duration, hanzi_slur
    
def parse_txt(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line) < 10:
                continue
            id, text, phoneme, note, note_duration, phoneme_duration, slur = line.split('|')
            id, text, initials, finals, note, note_duration, slur = parse_line(id, text, phoneme, note, note_duration, phoneme_duration, slur)
            data.append((id, text, initials, finals, note, note_duration, slur))
            # text not matter, just dor debug
    return data

def test_parse():
    id, text, initials, finals, note, note_duration, slur = parse_txt(DATASET_DIR+'/test.txt')[7]
    print(text)
    for i in range(len(initials)):
        print(initials[i],finals[i],'\t',slur[i],'\t',note_duration[i],'\t',note[i])

if __name__ == '__main__':
    DATASET_DIR = "opencpop"
    test_parse()