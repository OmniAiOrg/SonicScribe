from dataclasses import dataclass
from model.sonic_scriber import SonicScriber
from utils.general_dataloader import SonicBatch, WeightedDataset, WhisperDataCollatorWithPadding, dataset_keys
from utils.load_checkpoint import get_config
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from utils.general_dataloader import dataset_keys
import jiwer
class SonicLightling(LightningModule):
    def __init__(self, cfg, model:SonicScriber) -> None:
        super().__init__()
        self.model = model
        self.collate_fn = self.model.decoder.collate_fn
        self.all_tokenizers = self.collate_fn.all_tokenizers
        pad_label = self.collate_fn.word_tokenizer.pad_label
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
        
        self.wer = jiwer.wer
        def wer_compact(out_logits:dict, key:str, batch:SonicBatch,  output_cer:dict):
            logits = out_logits[key]
            label = batch.__getattribute__(f'{key}_label')
            # print(key, logits.shape,label.shape)
            tokenizer = self.all_tokenizers[key]
            logits = logits.argmax(dim=-1)
            label = [tokenizer.decode(label[i], compact=True, stop_at=tokenizer.eot) for i in range(self.cfg.batch_size)]
            pred = [tokenizer.decode(logits[i], compact=True, stop_at=tokenizer.pad) for i in range(self.cfg.batch_size)]
            compact_label = [label[i] for i in range(len(label)) if len(label[i])>0]
            compact_pred = [pred[i] for i in range(len(label)) if len(label[i])>0]
            if len(compact_label) == 0:
                wer_score = 0.0
            else:
                # print(key, '|', compact_pred[0], '|', compact_label[0])
                wer_score = self.wer(compact_label, compact_pred)
            output_cer[key] = wer_score
            output_cer['total'] += wer_score
            return wer_score
        self.wer_compact = wer_compact

        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        # self.model = torch.compile(self.model) # pytorch 2.0 surpport optimize model
    
    
    def forward(self, x:SonicBatch):
        return self.model.forward(x)
    
    def calculate_loss(self, out_logits, batch:SonicBatch, batch_id:int):
        out_loss = {'loss':0}
        self.ce_loss_compact(out_logits, 'hanzi', batch, out_loss)
        self.ce_loss_compact(out_logits, 'pinyin', batch, out_loss)
        self.ce_loss_compact(out_logits, 'note', batch, out_loss)
        self.ce_loss_compact(out_logits, 'tone', batch, out_loss)
        self.ce_loss_compact(out_logits, 'slur', batch, out_loss)
        self.ce_loss_compact(out_logits, 'start', batch, out_loss)
        self.ce_loss_compact(out_logits, 'end', batch, out_loss)
        return out_loss
    
    def calculate_char_error_rate(self, out_logits, batch:SonicBatch, batch_id:int):
        out_wer = {'total':0}
        self.wer_compact(out_logits, 'hanzi', batch, out_wer)
        self.wer_compact(out_logits, 'pinyin', batch, out_wer)
        self.wer_compact(out_logits, 'note', batch, out_wer)
        self.wer_compact(out_logits, 'tone', batch, out_wer)
        self.wer_compact(out_logits, 'slur', batch, out_wer)
        # self.cer_compact(out_logits, 'start', batch, out_cer)
        # self.cer_compact(out_logits, 'end', batch, out_cer)
        return out_wer

    def training_step(self, batch:SonicBatch, batch_id:int):
        out_logits = self.model(batch)
        out_loss = self.calculate_loss(out_logits, batch, batch_id)
        self.log("train/loss", out_loss['loss'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/word_loss", out_loss['hanzi'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/pinyin_loss", out_loss['pinyin'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/note_loss", out_loss['note'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/tone_loss", out_loss['tone'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/slur_loss", out_loss['slur'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/start_loss", out_loss['start'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("train/end_loss", out_loss['end'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return out_loss
    
    def validation_step(self, batch:SonicBatch, batch_id:int):
        with torch.no_grad():
            out_logits = self.model(batch)
            out_loss = self.calculate_loss(out_logits, batch, batch_id)
            out_wer = self.calculate_char_error_rate(out_logits, batch, batch_id)
            self.log("val_loss", out_loss['loss'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_word_loss", out_loss['hanzi'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_pinyin_loss", out_loss['pinyin'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_note_loss", out_loss['note'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_tone_loss", out_loss['tone'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_slur_loss", out_loss['slur'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_start_loss", out_loss['start'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_end_loss", out_loss['end'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            
            
            self.log("val_total_wer", out_wer['total'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_word_wer", out_wer['hanzi'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_pinyin_wer", out_wer['pinyin'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_note_wer", out_wer['note'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_tone_wer", out_wer['tone'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_slur_wer", out_wer['slur'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            return out_loss, out_wer
        
    def predict_step(self, batch, batch_id):
        with torch.no_grad():
            out_logits = self.model(batch)
            out_loss = self.calculate_loss(out_logits, batch, batch_id)
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

        # self.scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.cfg.warmup_steps, 
        #     num_training_steps=10
        # )
        
        torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)

        self.scheduler = torch_lr_scheduler

        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch", "frequency": 1}]
    
if __name__ == '__main__':
    all_config = get_config('data_reader/dataset_config.yaml')
    base_config = all_config['BaseReader']
    SAMPLE_RATE = base_config['SAMPLE_RATE']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    @dataclass
    class Config:
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
    collect_fn = WhisperDataCollatorWithPadding()
    model = SonicScriber('tiny', collate_fn=collect_fn)
    model = SonicLightling(cfg, model).to(DEVICE)
    
    from data_reader.opencpop import OpenCpop
    from data_reader.openslr_33 import Openslr33
    from torch.utils.data import DataLoader

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