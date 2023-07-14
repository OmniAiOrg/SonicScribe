import dataclasses
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class SonicData:
    mel: Tensor
    original_text: Optional[str] = None
    words: Optional[list[int]] = None
    note: Optional[list[int]] = None
    note_label: Optional[list[int]] = None
    note_duration: Optional[list[int]] = None
    note_duration_label: Optional[list[int]] = None
    slur: Optional[list[int]] = None
    slur_label: Optional[list[int]] = None
    initials: Optional[list[int]] = None
    initials_label: Optional[list[int]] = None
    finals: Optional[list[int]] = None
    finals_label: Optional[list[int]] = None
    
@dataclass
class SonicBatch:
    mel: Optional[Tensor] = None
    original_text: Optional[list[str]] = None
    words: Optional[Tensor] = None
    note: Optional[Tensor] = None
    note_label: Optional[Tensor] = None
    note_duration: Optional[Tensor] = None
    note_duration_label: Optional[Tensor] = None
    slur: Optional[Tensor] = None
    slur_label: Optional[Tensor] = None
    initials:Optional[Tensor] = None
    initials_label: Optional[Tensor] = None
    finals: Optional[Tensor] = None
    finals_label: Optional[Tensor] = None
    
non_tensor_key = ['original_text']
    
def sonic_batch_to_shape(sonic_batch: SonicBatch):
    field_names = [field.name for field in dataclasses.fields(SonicBatch)]
    sonic_batch_shape = {}
    for field_name in field_names:
        value = getattr(sonic_batch, field_name)
        if value is not None:
            sonic_batch_shape[field_name] = value.shape if field_name not in non_tensor_key else value
    return sonic_batch_shape
    

'''
Process list of SonicData to SonicBatch with padding.
In this call function, the input may be from multiple different dataset,
so a list may contain both None and not None values. 
'''
class WhisperDataCollatorWithPadding:
    def __init__(self, label_pad=-100, pad=3) -> None:
        self.const_pad_label = label_pad
        self.const_pad = pad
    
    def __call__(self, input: list[SonicData]) -> SonicBatch:
        sonic_batch = SonicBatch()
        field_names = [field.name for field in dataclasses.fields(SonicData)]
        features = {}
        for sonic_data in input:
            for field_name in field_names:
                if field_name not in features:
                    features[field_name] = []
                features[field_name].append(getattr(sonic_data, field_name))

        features['mel'] = torch.concat([m[None, :] for m in features['mel']])
        setattr(sonic_batch, 'original_text', features['original_text'])
        # setattr(sonic_batch, 'words', features['words']) if not all(element is None for element in features['words'])
        # setattr(sonic_batch, 'mel', mel)
        
        feature_lengths = [len(p) if p is not None else 0 for p in features['initials']]
        max_feature_len = max(feature_lengths)

        
        for k, v in features.items():
            if k in non_tensor_key + ['mel']:
                continue
            if 'label' in k:
                constant_values = self.const_pad_label
            else:
                constant_values = self.const_pad
            if all(element is None for element in v):
                continue
            feature_lengths_ = [len(p) if p is not None else 0 for p in v]
            assert feature_lengths_ == feature_lengths, f'{feature_lengths_}, {feature_lengths}'
            features[k] = [np.pad(f, (0, max_feature_len - f_len), 'constant', constant_values=constant_values) 
                            for f, f_len in zip(v, feature_lengths)]

        [setattr(sonic_batch, field_name, torch.tensor(np.array(v))) for field_name, v in features.items() if field_name not in non_tensor_key]

        return sonic_batch
    
if __name__ == '__main__':
    from data_reader.opencpop import OpenCpop
    dataset = OpenCpop(train=False)
    loader = torch.utils.data.DataLoader(dataset, 
                          batch_size=4, 
                          collate_fn=WhisperDataCollatorWithPadding()
                          )
    for b in loader:
        print(sonic_batch_to_shape(b))
        break