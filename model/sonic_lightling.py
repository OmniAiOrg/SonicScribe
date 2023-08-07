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
from pytorch_lightning.loggers import TensorBoardLogger
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
            output_loss['total'] += loss_out
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
                logger:TensorBoardLogger = self.logger
                log_str = ''
                for i in range(len(compact_pred)):
                    log_str += f'**pred**:  {compact_pred[i]}  \n'
                    log_str += f'**label**: {compact_label[i]}  \n'
                logger.experiment.add_text(f'wer/{key}', log_str, self.global_step)
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
    
    def calculate_loss(self, out_logits, batch:SonicBatch, batch_id:int, keys):
        out_loss = {'total':0}
        calculate = lambda key: self.ce_loss_compact(out_logits, key, batch, out_loss)
        for key in keys:
            calculate(key)
        return out_loss
    
    def calculate_char_error_rate(self, out_logits, batch:SonicBatch, batch_id:int):
        out_wer = {'total':0}
        calculate = lambda key: self.wer_compact(out_logits, key, batch, out_wer)
        for key in ['hanzi', 'pinyin', 'note', 'tone', 'slur']:
            calculate(key)
        return out_wer

    def training_step(self, batch:SonicBatch, batch_id:int):
        out_logits = self.model(batch)
        keys = ['hanzi', 'pinyin', 'tone']#, 'note', 'slur', 'start', 'end']
        out_loss = self.calculate_loss(out_logits, batch, batch_id, keys)
        self.log(f'train/loss', out_loss['total'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        save_loss = lambda key : self.log(f'train/{key}_loss', out_loss[key], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        for key in keys:
            save_loss(key)
        out_loss['loss'] = out_loss['total']
        return out_loss
    
    def validation_step(self, batch:SonicBatch, batch_id:int):
        out_logits = self.model(batch)
        keys = ['hanzi', 'pinyin', 'tone']#, 'note', 'slur', 'start', 'end']
        out_loss = self.calculate_loss(out_logits, batch, batch_id, keys)
        self.log(f'val/loss', out_loss['total'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        save_loss = lambda key: self.log(f'val/{key}_loss', out_loss[key], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        for key in keys:
            save_loss(key)
        out_loss['loss'] = out_loss['total']

        out_wer = self.calculate_char_error_rate(out_logits, batch, batch_id)
        save_wer = lambda key: self.log(f"wer/{key}", out_wer[key], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        for key in ['total', 'hanzi', 'pinyin', 'note', 'tone', 'slur']:
            save_wer(key)
        return out_loss, out_wer
        
    def predict_step(self, batch, batch_id):
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