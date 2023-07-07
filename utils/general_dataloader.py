import torch
import numpy as np

class WhisperDataCollatorWithPadding:
    def __init__(self, label_pad=-100, pad=3) -> None:
        self.const_pad_label = label_pad
        self.const_pad = pad
    
    def __call__(self, input):
        features = {}
        for f in input:
            for k, v in f.items():
                if k not in features:
                    features[k] = []
                features[k].append(v)

        features['mel'] = torch.concat([m[None, :] for m in features['mel']])
        text = features['text']
        
        feature_lengths = [len(p) for p in features['initials']]
        max_feature_len = max(feature_lengths)

        for k, v in features.items():
            if k in ['mel', 'text']:
                continue
            if 'label' in k:
                constant_values = self.const_pad_label
            else:
                constant_values = self.const_pad
                # constant_values = -100
            features[k] =  [np.pad(f, (0, max_feature_len - f_len), 'constant', constant_values=constant_values) for f, f_len in zip(v, feature_lengths)]

        features = {k: torch.tensor(np.array(v)) for k, v in features.items() if k not in ['text']}
        features['text'] = text

        return features