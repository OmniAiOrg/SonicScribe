from dataclasses import dataclass
from model.whisper_official import WhisperOfficial
from utils.general_dataloader import SonicBatch, WhisperOfficialBatch, WhisperOfficialDataCollatorWithPadding, dataset_keys
from utils.load_checkpoint import get_config
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch import Tensor, nn
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import jiwer

from utils.naive_tokenizer import WhisperTokenizer
class WhisperOfficialLightling(LightningModule):
    def __init__(self, cfg, model:WhisperOfficial) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = self.model.tokenizer
        pad_label = self.tokenizer.pad
        self.dataset_keys = dataset_keys
        self.batch_size = cfg.batch_size
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_label)
        
        self.wer = jiwer.wer
        def wer_compact(out_logits:dict, batch:WhisperOfficialBatch):
            key = 'data'
            logits = out_logits
            label = batch.__getattribute__(f'{key}_label')
            # print(key, logits.shape,label.shape)
            tokenizer:WhisperTokenizer = self.tokenizer
            logits = logits.argmax(dim=-1)
            label = [tokenizer.decode(label[i], stop_at=tokenizer.eot) for i in range(self.cfg.batch_size)]
            pred = [tokenizer.decode(logits[i], stop_at=tokenizer.eot) for i in range(self.cfg.batch_size)]
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
            return wer_score
        self.wer_compact = wer_compact

        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        # self.model = torch.compile(self.model) # pytorch 2.0 surpport optimize model
    
    
    def forward(self, x:WhisperOfficialBatch):
        return self.model.forward(x)
    
    def calculate_loss(self, out_logits:Tensor, batch:WhisperOfficialBatch, batch_id:int):
        out_logits = torch.transpose(out_logits, 1, 2)
        # print(out_logits.shape, batch.data_label.shape)
        out_loss = self.ce_loss(out_logits, batch.data_label)
        return out_loss
    
    def calculate_char_error_rate(self, out_logits, batch:WhisperOfficialBatch, batch_id:int):
        out_wer = self.wer_compact(out_logits, batch)
        return out_wer

    def training_step(self, batch:WhisperOfficialBatch, batch_id:int):
        out_logits = self.model(batch)
        loss = self.calculate_loss(out_logits, batch, batch_id)
        self.log(f'train/loss', loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch:WhisperOfficialBatch, batch_id:int):
        out_logits = self.model(batch)
        loss = self.calculate_loss(out_logits, batch, batch_id)
        out_wer = self.calculate_char_error_rate(out_logits, batch, batch_id)
        self.log(f'val/loss', loss, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log(f'val/wer', out_wer, on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss, out_wer
        
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
    model = WhisperOfficial('tiny')
    model = WhisperOfficialLightling(cfg, model).to(DEVICE)
    
    from data_reader.opencpop import OpenCpop
    from torch.utils.data import DataLoader
    dataset_a = OpenCpop(train=False)
    loader = DataLoader(dataset_a, 
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            collate_fn=WhisperOfficialDataCollatorWithPadding(),
                            pin_memory=True,
                            pin_memory_device = DEVICE
                          )
    trainer.fit(model, loader)