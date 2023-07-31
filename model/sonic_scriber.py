'''
SonicScriber is a modified whisper model that can transcribe singing to text.
The model architecture is similar with whisper, with different tokenizer 
and multiple task loss.
This project is a base project for singing generation--Only with bounch of 
data a 
'''

import os
import numpy as np
from utils.general_dataloader import SonicBatch, WeightedDataset, WhisperDataCollatorWithPadding, get_field_names, dataset_keys
from utils.load_checkpoint import get_config, get_whisper_checkpoint

import torch
from torch import nn
# import pandas as pd
# import importlib
import whisper
# importlib.reload(whisper)
import torchaudio
import torchaudio.transforms as at
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm.notebook import tqdm
import pickle
from typing import Dict, List, Optional, Tuple
from functools import cached_property
from transformers import (
    get_linear_schedule_with_warmup,
    # AdamW
)
from whisper.model import Whisper, ModelDimensions, AudioEncoder
from torch import Tensor
from torch.nn import LayerNorm
from whisper.model import ResidualAttentionBlock, TextDecoder, Iterable, ModelDimensions
from typing import Optional

class CustomTextDecoder(nn.Module):
    def __init__(
        self, dims:ModelDimensions, collate_fn: WhisperDataCollatorWithPadding
    ):
        super().__init__()
        n_state = dims.n_text_state
        n_head = dims.n_text_head
        n_ctx = dims.n_text_ctx
        n_layer = dims.n_text_layer
        
        self.collate_fn = collate_fn
        
        self.word_embedding = collate_fn.word_embedding.chinese_token_embedding
        self.pinyin_embedding = collate_fn.pinyin_embedding.pinyin_token_embedding
        self.tone_token_embedding = nn.Embedding(len(collate_fn.tone_tokenizer.vocabs)+1, n_state)
        self.note_token_embedding = nn.Embedding(len(collate_fn.note_tokenizer.vocabs)+1, n_state)
        self.duration_token_embedding = nn.Embedding(len(collate_fn.duratoin_tokenizer.vocabs)+1, n_state)
        self.slur_token_embedding = nn.Embedding(len(collate_fn.slur_tokenizer.vocabs)+1, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
        torch.nn.init.xavier_uniform_(self.positional_embedding)

        # self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
        #     [copy.deepcopy(block) for block in sample_decoder.blocks[:-1]]
        # )
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer - 1) # 减1，去掉最后一层
            ]
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        
        self.filed_names = [i for i in dataset_keys if i not in ['audio']]
        print('filed_names', self.filed_names)
        self.head_ln = {}
        self.head_block = {}
        for field_name in self.filed_names:
            self.head_ln[field_name] = LayerNorm(n_state)
            self.head_block[field_name] = ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            self.head_block[field_name].apply(init_weights)
        
        # self.dur_mlp = nn.Sequential(
        #     nn.Linear(n_state, 30),
        #     nn.ReLU(),
        #     nn.Linear(30, 1)
        # )
        # # self.slur_mlp = nn.Linear(n_state, 1)
        # self.dur_mlp.apply(init_weights)
        # # self.slur_mlp.apply(init_weights)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        
    def forward(self, batch:SonicBatch, audio_embedding: Tensor, kv_cache: Optional[dict] = None) -> dict:
        """
        batch : features
        embedding : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        
        def get_embedding(input:Tensor, embedding: nn.Embedding):
            output = (
                embedding(input)
                + self.positional_embedding[offset : offset + input.shape[-1]]
            )
            return output.to(audio_embedding.dtype)
            
        word = get_embedding(batch.hanzi, self.word_embedding)
        pinyin = get_embedding(batch.pinyin, self.pinyin_embedding)
        note = get_embedding(batch.note, self.note_token_embedding)
        tone = get_embedding(batch.tone, self.tone_token_embedding)
        slur = get_embedding(batch.slur, self.slur_token_embedding)

        feature_embedding = word + pinyin + note + tone + slur

        # with torch.no_grad():
        for block in self.blocks:
            feature_embedding = block(feature_embedding, audio_embedding, mask=self.mask, kv_cache=kv_cache)
            
        head_out = {}
        for field_name in self.filed_names:
            head_out[field_name] = self.head_ln[field_name](self.head_block[field_name](feature_embedding, audio_embedding, mask=self.mask, kv_cache=kv_cache))

        output_logits = {}
        def get_logits(field_name: str, embedding: nn.Embedding):
            output_logits[field_name] = (
                head_out[field_name] @ torch.transpose(embedding.weight, 0, 1)
            )
            return output_logits[field_name]
    
        get_logits('hanzi', self.word_embedding)
        get_logits('pinyin', self.pinyin_embedding)
        get_logits('note', self.note_token_embedding)
        get_logits('tone', self.tone_token_embedding)
        get_logits('slur', self.slur_token_embedding)
        get_logits('start', self.duration_token_embedding)
        get_logits('end', self.duration_token_embedding)
        
        return output_logits

class SonicScriber(Whisper):
    def get_model_dims(self, model='tiny'):
        checkpoint = get_whisper_checkpoint(model)   
        dims = ModelDimensions(**checkpoint["dims"])
        return dims

    def __init__(self, model='tiny', collate_fn=WhisperDataCollatorWithPadding()):
        dims = self.get_model_dims(model)
        print(dims)
        super().__init__(dims)
        self.decoder = CustomTextDecoder(dims, collate_fn)
        
    def forward(self, batch:SonicBatch) -> Dict[str, Tensor]:
        return self.decoder(batch, self.encoder(batch.mel))
        
if __name__ == '__main__':
    from torch.utils.data import ConcatDataset, DataLoader
    print('torch=', torch.__version__, 'lightling=', pl.__version__)
    config = get_config()
    SEED = int(config['trainer']['seed'])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    seed_everything(SEED, workers=True)
    collate_fn=WhisperDataCollatorWithPadding()
    sonic_scriber = SonicScriber('tiny', collate_fn=collate_fn)
    
    from data_reader.opencpop import OpenCpop
    from data_reader.openslr_33 import Openslr33
    dataset_a = OpenCpop(train=False)
    dataset_b = Openslr33(train=False)
    print(len(dataset_a), len(dataset_b), flush=True)
    weighted_dataset = WeightedDataset([(dataset_a, 3), (dataset_b, 3)])
    
    loader = DataLoader(weighted_dataset, 
                            batch_size=10,
                            shuffle=True,
                            collate_fn=collate_fn
                          )
    for b in loader:
        # print(b)
        logits = sonic_scriber.forward(b)
        for k, v in logits.items():
            print(f'{k:<8}', v.shape)
        break
    
