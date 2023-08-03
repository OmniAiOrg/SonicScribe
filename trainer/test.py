from dataclasses import dataclass
from pathlib import Path
from model.sonic_lightling import SonicLightling
from model.sonic_scriber import SonicScriber
from utils.general_dataloader import WeightedDataset, WhisperDataCollatorWithPadding
from utils.load_checkpoint import get_config
from data_reader.opencpop import OpenCpop
from data_reader.openslr_33 import Openslr33
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.model_weight_initializer import initialize_sonic_scriber_from_whisper

# 1. training settings
all_config = get_config('data_reader/dataset_config.yaml')
data_config = all_config['BaseReader']

SAMPLE_RATE = data_config['SAMPLE_RATE']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
train_id = "001"
log_output_dir = "./logs"
check_output_dir = "./artifacts"
model_size = "tiny"
train_name = "SonicScribe"

@dataclass
class Config:
    learning_rate = 0.001
    weight_decay = 0.01
    adam_epsilon = 1e-7
    warmup_steps = 2
    batch_size = 8
    precision = '16-mixed' # 32 for single, 16 for half (faster)
    num_worker = 8 # <= batch_size, <= cpu cores, larger is better
    pin_memory=True
    num_train_epochs = 40
    gradient_accumulation_steps = 3
    sample_rate = SAMPLE_RATE
    limit_val_batches = 0.25
    overfit_batches = 0 # 0 by default. Set to 0.005 for overfit sanity check
    log_every_n_steps = 1
    
# 2. Trainer preparation

Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint/{train_id}",
    filename="checkpoint-{epoch:03d}-{val/loss:.2f}",
    save_top_k=5,
    monitor = 'val/loss',
    auto_insert_metric_name=False,
    save_last=True,
)

latest_ckpt = Path(f"{check_output_dir}/checkpoint/last.ckpt")
if latest_ckpt.is_file():
    ckpt_path = latest_ckpt
else:
    ckpt_path = None
    
callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

cfg = Config()
trainer = Trainer(
    accelerator=DEVICE,
    logger=tflogger,
    callbacks=callback_list,
    precision=cfg.precision,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    limit_val_batches=cfg.limit_val_batches,
    overfit_batches=cfg.overfit_batches,
    log_every_n_steps = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.log_every_n_steps
    )
collect_fn = WhisperDataCollatorWithPadding(model = model_size, num_workers = cfg.num_worker)
model = SonicScriber(model_size, collate_fn=collect_fn)
initialize_sonic_scriber_from_whisper(model)
model = SonicLightling(cfg, model=model).to(DEVICE)

# 3. Dataset for training and validation
def get_dataloader(train=True) -> DataLoader:
    dataset_a = OpenCpop(train)
    dataset_b = Openslr33(train)
    print('Train' if train else 'Val', len(dataset_a), len(dataset_b), flush=True)
    weighted_dataset = WeightedDataset([(dataset_a, 3), (dataset_b, 3)])
    dataloader = DataLoader(weighted_dataset, 
                            batch_size=cfg.batch_size,
                            shuffle=True if train and cfg.overfit_batches==0 else False,
                            collate_fn=collect_fn,
                            num_workers = cfg.num_worker,
                            pin_memory=True,
                            pin_memory_device = DEVICE
                            )
    return dataloader
    
train_dataloader = get_dataloader(True)
val_dataloader = get_dataloader(False)

# 4. test the model
trainer.validate(model, val_dataloader)