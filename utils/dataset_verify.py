'''
First, obtain the dataset, then sample some data and save it to
the 'image' in the text file. This is necessary because the dataset 
may contain small errors that could disrupt the training process.
'''

import dataclasses
from data_reader.opencpop import OpenCpop
from utils.general_dataloader import SonicData
from utils.tokenizers import *

def sonic_data_to_image(data: SonicData):
    image = ''
    field_names = [field.name for field in dataclasses.fields(SonicData) 
                   if getattr(data, field.name) is not None 
                   and field.name not in ['mel', 'note_duration', 'original_text']]
    for field_name in field_names:
        # assert f'{field_name}_tokenizer' in globals(), f'{field_name}_tokenizer'
        # tokenizer = 
        image += f'\n{field_name[:10]}:\t'
        image += '|'.join([str(
            globals().get(f'{field_name}_tokenizer').decode([i])[:5] if field_name != 'original_text' else i
            )+'\t' for i in getattr(data, field_name)])
        # image += '|'.join(str(getattr(data, field_name)))
    return image

def opencpop_verify():
    oc_train = OpenCpop()
    oc_test = OpenCpop(train=False)
    data = oc_test[0]
    print(sonic_data_to_image(data))
    
    
if __name__ == '__main__':
    opencpop_verify()