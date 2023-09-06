from dataclasses import dataclass
from pathlib import Path
from data_reader.openslr_47 import Openslr47
from data_reader.openslr_68 import Openslr68
from model.sonic_lightling import SonicLightling
from model.sonic_scriber import SonicScriber
from model.whisoer_official_lightling import WhisperOfficialLightling
from model.whisper_official import WhisperOfficial
from utils.general_dataloader import WeightedDataset, WhisperDataCollatorWithPadding, WhisperOfficialDataCollatorWithPadding
from utils.load_checkpoint import get_config
from data_reader.opencpop import OpenCpop
from data_reader.openslr_33 import Openslr33
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.model_weight_initializer import initialize_sonic_scriber_from_whisper, initialize_whisper_official_from_checkpoint, initialize_whisper_official_from_whisper

print('torch=', torch.__version__, 'torch_lightling=', pl.__version__)
# 1. training settings
all_config = get_config('data_reader/dataset_config.yaml')
data_config = all_config['BaseReader']

SAMPLE_RATE = data_config['SAMPLE_RATE']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_id = "whisper_official_timestamp_007"
log_output_dir = "./logs"
check_output_dir = "./artifacts"
model_size = "tiny"
train_name = "WhisperOfficial"
# resume_checkpoint = "whisper_official_005/checkpoint-002-0.00.ckpt"
resume_checkpoint = "whisper_official_timestamp_006/last-v1.ckpt"

@dataclass
class Config:
    learning_rate = 0.001
    weight_decay = 0.01
    adam_epsilon = 1e-7
    warmup_steps = 2
    batch_size = 6
    precision = '16-mixed' # 32 for single, 16 for half (faster)
    num_worker = 8 # <= batch_size, <= cpu cores, larger is better
    pin_memory=True
    num_train_epochs = 50
    gradient_accumulation_steps = 4
    sample_rate = SAMPLE_RATE
    limit_val_batches = 1
    overfit_batches = 0 # 0 by default. Set to 0.005 for overfit sanity check
    log_every_n_steps = 1
    val_check_interval = None # None for default, set to 2000 here. Even not end of epoch, run validation step every these amount of steps
    num_sanity_val_steps = 10 # 1 by default
    enable_progress_bar = True # False for nohup
    
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
    every_n_train_steps=1000
)

latest_ckpt = Path(f"{check_output_dir}/checkpoint/{resume_checkpoint}")
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
    log_every_n_steps = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.log_every_n_steps,
    val_check_interval = cfg.val_check_interval,
    num_sanity_val_steps = cfg.num_sanity_val_steps,
    enable_progress_bar = cfg.enable_progress_bar
    )
collect_fn = WhisperOfficialDataCollatorWithPadding(model = model_size, num_workers = cfg.num_worker)
model = WhisperOfficial('tiny')
if ckpt_path == None:
    initialize_whisper_official_from_whisper(model)
else:
    initialize_whisper_official_from_checkpoint(ckpt_path, model)
model = WhisperOfficialLightling(cfg, model).to(DEVICE)

# 3. Dataset for training and validation
def naive_dataloader(dataset, train):
    return  DataLoader(dataset, 
                batch_size=cfg.batch_size,
                shuffle=True if train and cfg.overfit_batches==0 else False,
                collate_fn=collect_fn,
                num_workers = cfg.num_worker,
                pin_memory=True,
                pin_memory_device = DEVICE,
                persistent_workers=True
                )

def get_dataloader(train=True) -> DataLoader:
    dataset_a = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'note'])
    dataset_b = OpenCpop(train=train, key_filter=['audio', 'hanzi']) # hanzi only
    dataset_c = OpenCpop(train=train, key_filter=['audio', 'note']) # note only
    dataset_d = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'note', 'end'])
    dataset_e = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'end']) # hanzi only
    dataset_f = OpenCpop(train=train, key_filter=['audio', 'note', 'end']) # note only
    weighted_dataset = WeightedDataset([
        (dataset_a, 1, 'order'), (dataset_a, 1), 
         (dataset_b, 1), (dataset_b, 1, 'order'), 
         (dataset_c, 1), (dataset_c, 1, 'order'), 
         (dataset_d, 1),
         (dataset_d, 1, 'order'), 
         (dataset_e, 1),(dataset_e, 1, 'order'), 
         (dataset_f, 1), (dataset_f, 1, 'order')
         ])
    dataloader = naive_dataloader(weighted_dataset, train)
    return dataloader
    
train_dataloader = get_dataloader(True)

def get_val_dataloader(train=True) -> DataLoader:
    # dataset_a = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'note'])
    dataset_b = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'note', 'end']) # hanzi only
    weighted_dataset = WeightedDataset([
        # (dataset_a, 1, 'order'), (dataset_a, 1), 
        #  (dataset_b, 1), 
         (dataset_b, 1, 'order')])
    dataloader = naive_dataloader(weighted_dataset, train)
    return dataloader
val_dataloader = get_val_dataloader(False)

# 4. fit the model
trainer.fit(
    model, 
    train_dataloader, 
    val_dataloader,
    # ckpt_path = ckpt_path
    )