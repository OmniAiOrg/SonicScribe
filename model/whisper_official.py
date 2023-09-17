import numpy as np
from utils.general_dataloader import SonicBatch, WeightedDataset, WhisperDataCollatorWithPadding, WhisperOfficialBatch, WhisperOfficialDataCollatorWithPadding, dataset_keys
from utils.load_checkpoint import get_config, get_whisper_checkpoint

import torch

from whisper.model import Whisper, ModelDimensions
from torch import Tensor
from whisper.model import ModelDimensions, TextDecoder
from typing import Dict, Iterable, Optional
from utils.naive_tokenizer import WhisperTokenizer, get_tokenizer

class WhisperOfficial(Whisper):
    def get_model_dims(self, model='tiny'):
        checkpoint = get_whisper_checkpoint(model)   
        dims = ModelDimensions(**checkpoint["dims"])
        assert dims.n_vocab == 51865
        self.tokenizer: WhisperTokenizer = get_tokenizer(multilingual=True, language='zh', task='transcibe')
        # n_vocab = self.tokenizer.encode('<|pad|>', allowed_special="all")[0]+1
        n_vocab = self.tokenizer.encoding.n_vocab
        assert n_vocab >= dims.n_vocab, f'n_vocab={n_vocab} dims.n_vocab={dims.n_vocab}'
        dims.n_vocab = n_vocab
        return dims

    def __init__(self, model='tiny', stop_grad_on_encoder=False):
        self.model_size = model
        self.stop_grad_on_encoder = stop_grad_on_encoder
        dims = self.get_model_dims(model)
        super().__init__(dims)
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        if stop_grad_on_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    @property
    def is_multilingual(self):
        return True
    
    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.stop_grad_on_encoder:
            with torch.no_grad():
                audio_encoded = self.encoder(mel)
            return self.decoder(tokens, audio_encoded)
        else:
            return self.decoder(tokens, self.encoder(mel))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    print('torch=', torch.__version__)
    config = get_config()
    SEED = int(config['trainer']['seed'])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    whisper_official = WhisperOfficial('tiny').to(DEVICE)
    
    with open('./assets/whisper_official_model.txt', 'w') as f:
        f.write(str(whisper_official))
        
    from data_reader.opencpop import OpenCpop
    dataset_a = OpenCpop(train=False)
    
    loader = DataLoader(dataset_a, 
                            batch_size=10,
                            shuffle=True,
                            collate_fn=WhisperOfficialDataCollatorWithPadding(),
                            pin_memory=True,
                            pin_memory_device = DEVICE
                          )
    for b in loader:
        b:WhisperOfficialBatch = b
        embeddings = whisper_official.encoder.forward(b.mel)
        logits = whisper_official.decoder.forward(b.data, embeddings)
        # logits = whisper_official.forward(batch=b)
        print(logits.shape)
        print(b.mel.shape, embeddings.shape)
        break
    