from dataclasses import dataclass
import os
import numpy as np
from model.sonic_scriber import SonicScriber
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
from utils.general_dataloader import dataset_keys

class SonicLightling(LightningModule):
    def __init__(self, cfg, model="tiny", lang="zh", device='cpu') -> None:
        super().__init__()
        self.model = SonicScriber(model).to(device=device)
        # self.metrics_wer = evaluate.load("wer")
        # self.metrics_cer = evaluate.load("cer")
        pad_label = self.model.decoder.collate_fn.word_tokenizer.pad_label
        self.dataset_keys = dataset_keys
        self.batch_size = cfg.batch_size
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_label)
        self.sooth_ce_loss = nn.CrossEntropyLoss(ignore_index=pad_label, label_smoothing=0.2)
        slur_tokenizer = self.model.decoder.collate_fn.slur_tokenizer
        slur_weight = len(slur_tokenizer.vocabs) * [1]
        slur_weight[slur_tokenizer.converter['1']] = 10
        self.slur_loss = nn.CrossEntropyLoss(ignore_index=pad_label, weight=torch.FloatTensor(slur_weight))
        
        def ce_loss_compact(out_logits:dict, key:str, batch:SonicBatch, output_loss:dict):
            assert key in dataset_keys
            if key in ['start', 'end']:
                loss = self.sooth_ce_loss
            elif key == 'slur':
                loss = self.slur_loss
            else:
                loss = self.ce_loss
            index = dataset_keys.index(key)
            mask = batch.mask[:,index].unsqueeze(1)
            a = out_logits[key].transpose(1, 2)
            b = batch.__getattribute__(f'{key}_label')*mask
            loss_out = loss(a, b)
            output_loss[key] = loss_out
            output_loss['loss'] += loss_out
            return loss_out
        self.ce_loss_compact = ce_loss_compact
        
        
        def weighted_ce_loss(predict, label):
            predict = predict.view(-1, predict.size(-1))
            label = label.view(-1)
            return self.slur_loss(predict, label)
        
        self.wce_loss = weighted_ce_loss

        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        # self.model = torch.compile(self.model) # pytorch 2.0 surpport optimize model
    
    
    def forward(self, x:SonicBatch):
        return self.model.forward(x)
    
    def calculate_loss(self, batch:SonicBatch, batch_id:int):
        out_logits = self.forward(batch)
        out_loss = {'loss':0}
        word_loss = self.ce_loss_compact(out_logits, 'hanzi', batch, out_loss)
        pinyin_loss = self.ce_loss_compact(out_logits, 'pinyin', batch, out_loss)
        note_loss = self.ce_loss_compact(out_logits, 'note', batch, out_loss)
        tone_loss = self.ce_loss_compact(out_logits, 'tone', batch, out_loss)
        slur_loss = self.ce_loss_compact(out_logits, 'slur', batch, out_loss)
        start_loss = self.ce_loss_compact(out_logits, 'start', batch, out_loss)
        end_loss = self.ce_loss_compact(out_logits, 'end', batch, out_loss)
        return out_loss

    def training_step(self, batch:SonicBatch, batch_id:int):
        out_loss = self.calculate_loss(batch, batch_id)
        self.log("train/loss", out_loss['loss'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/word_loss", out_loss['hanzi'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/note_loss", out_loss['note'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/tone_loss", out_loss['tone'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/slur_loss", out_loss['slur'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return out_loss
    
    def validation_step(self, batch:SonicBatch, batch_id:int):
        with torch.no_grad():
            out_loss = self.calculate_loss(batch, batch_id)
            self.log("val/loss", out_loss['loss'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val/word_loss", out_loss['hanzi'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val/note_loss", out_loss['note'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val/tone_loss", out_loss['tone'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val/slur_loss", out_loss['slur'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            return out_loss
        
    def predict_step(self, batch, batch_id):
        with torch.no_grad():
            out_loss = self.calculate_loss(batch, batch_id)
            return out_loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # do_not_update_params = ['decoder.blocks', 'encoder']
        do_not_update_params = []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if not any(nd in n for nd in no_decay)
                            and not any((nd in n) for nd in do_not_update_params)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if any(nd in n for nd in no_decay)
                            and not any((nd in n) for nd in do_not_update_params)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                          lr=self.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=10
        )

        return [optimizer], [{"scheduler": self.scheduler, "interval": "step", "frequency": 1}]
    
    # def setup(self, stage=None):
        # if stage == 'fit' or stage is None:
        #     self.t_total = (
        #         (len(self.__train_dataset) // (self.cfg.batch_size))
        #         // self.cfg.gradient_accumulation_steps
        #         * float(self.cfg.num_train_epochs)
        #     )
    
if __name__ == '__main__':
    all_config = get_config('data_reader/dataset_config.yaml')
    base_config = all_config['BaseReader']
    SAMPLE_RATE = base_config['SAMPLE_RATE']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    @dataclass
    class Config:
        # learning_rate = 1.1306633979496386e-06
        learning_rate = 0.01
        weight_decay = 0.01
        adam_epsilon = 1e-7
        warmup_steps = 2
        batch_size = 8
        num_worker = 8
        pin_memory=True
        num_train_epochs = 40
        gradient_accumulation_steps = 1
        sample_rate = SAMPLE_RATE
        
    cfg = Config()
    trainer = Trainer(accelerator=DEVICE)
    model = SonicLightling(cfg).to(DEVICE)
    
    from data_reader.opencpop import OpenCpop
    from data_reader.openslr_33 import Openslr33
    from torch.utils.data import DataLoader
    collect_fn = WhisperDataCollatorWithPadding()
    dataset_a = OpenCpop(train=True)
    dataset_b = Openslr33(train=True)
    print(len(dataset_a), len(dataset_b), flush=True)
    weighted_dataset = WeightedDataset([(dataset_a, 3), (dataset_b, 3)])
    
    train_dataloader = DataLoader(weighted_dataset, 
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            collate_fn=collect_fn,
                            num_workers = 8,
                            pin_memory=True,
                            pin_memory_device = DEVICE
                          )
    
    trainer.fit(model, train_dataloader)