from dataclasses import dataclass
from pathlib import Path
from data_reader.openslr_38 import Openslr38
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
from utils.preparation.human_dataloader_verify import verify_dataloader_by_human
from utils.whisper_dataloader import OpenCpopDataCollatorWithPadding, RandomReplaceDataset, SpeechDataCollatorWithPadding, WrappedDataset, mixedDataCollator

print('torch=', torch.__version__, 'torch_lightling=', pl.__version__)
# 1. training settings
all_config = get_config('data_reader/dataset_config.yaml')
data_config = all_config['BaseReader']

SAMPLE_RATE = data_config['SAMPLE_RATE']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_id = "opencpop_010"
log_output_dir = "./logs"
check_output_dir = "./artifacts"
model_size = "tiny"
train_name = "WhisperOfficial"
resume_checkpoint = "opencpop_009/last.ckpt"

@dataclass
class Config:
    learning_rate = 1e-5 #5.281682335805869e-05
    weight_decay = 1e-3
    adam_epsilon = 1e-7
    warmup_steps = 0
    batch_size = 8
    precision = '16-mixed' # 32 for single, 16 for half (faster)
    num_worker = 8 # = cpu cores
    pin_memory=True
    num_train_epochs = 50
    gradient_accumulation_steps = 3
    sample_rate = SAMPLE_RATE
    overfit_batches = 0 # 0 by default. Set to 0.005 for overfit sanity check
    log_every_n_steps = 1
    limit_val_batches = 0.02 # 0.02 when train, None else
    val_check_interval = 2000 # None for default, set to 2000 here. Even not end of epoch, run validation step every these amount of steps
    num_sanity_val_steps = 10 # 1 by default
    enable_progress_bar = True # False for nohup
    stop_grad_on_encoder = True
    num_concat = 4
    
# 2. Trainer preparation
cfg = Config()
Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint/{train_id}",
    filename="checkpoint-{epoch:03d}-{val/loss:.5f}",
    save_top_k=5,
    monitor = 'val/loss',
    auto_insert_metric_name=False,
    save_last=True,
    every_n_train_steps=cfg.val_check_interval if cfg.val_check_interval is not None else 1000
)

latest_ckpt = Path(f"{check_output_dir}/checkpoint/{resume_checkpoint}")
if latest_ckpt.is_file() and not cfg.overfit_batches>0:
    ckpt_path = latest_ckpt
else:
    ckpt_path = None
    
callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

trainer = Trainer(
    accelerator=DEVICE,
    logger=tflogger,
    callbacks=callback_list,
    precision=cfg.precision,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    overfit_batches=cfg.overfit_batches,
    limit_val_batches=cfg.limit_val_batches,
    log_every_n_steps = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.log_every_n_steps,
    val_check_interval = cfg.val_check_interval,
    num_sanity_val_steps = cfg.num_sanity_val_steps,
    enable_progress_bar = cfg.enable_progress_bar,
    profiler="simple" if cfg.overfit_batches > 0 else None
    )
# collect_fn = SpeechDataCollatorWithPadding(model = model_size, num_workers = cfg.num_worker, num_concat=cfg.num_concat)
model = WhisperOfficial('tiny', stop_grad_on_encoder=cfg.stop_grad_on_encoder)
if ckpt_path == None:
    initialize_whisper_official_from_whisper(model)
    log_path = log_output_dir+'/'+train_name+'/'+train_id
    print('clean log path', log_path)
    for item in Path(log_path).glob('*'):
        if item.is_file():
            item.unlink()  # rm file
else:
    initialize_whisper_official_from_checkpoint(ckpt_path, model)
model = WhisperOfficialLightling(cfg, model).to(DEVICE)

# 3. Dataset for training and validation
def naive_dataloader(dataset, collect_fn, train):
    return  DataLoader(dataset, 
                batch_size=cfg.batch_size,
                shuffle=True if train and cfg.overfit_batches==0 else False,
                collate_fn=collect_fn,
                num_workers = cfg.num_worker if train else 4,
                pin_memory=cfg.pin_memory,
                pin_memory_device = DEVICE if cfg.pin_memory else "",
                persistent_workers=True
                )

def get_dataloader(train=True) -> DataLoader:
    dataset_a = Openslr38(train=train, key_filter=['audio', 'hanzi', 'waveform'])
    dataset_b = Openslr33(train=train, key_filter=['audio', 'hanzi', 'waveform'])
    dataset_c = Openslr47(train=train, key_filter=['audio', 'hanzi', 'waveform'])
    dataset_d = Openslr68(train=train, key_filter=['audio', 'hanzi', 'waveform'])
    openslr = WrappedDataset([
        (dataset_a, 1, []),
        (dataset_b, 1, []),
        (dataset_c, 1, []),
        (dataset_d, 1, []),
        ])
    openCpop = OpenCpop(train=train, key_filter=['audio', 'hanzi', 'note', 'start', 'end', 'waveform'])
    wrapped_dataset = RandomReplaceDataset([
        # (openslr, 1, ['notimestamp', 'pad']),
        (openslr, 0.4, ['notimestamp', 'order', 'cluster:speech', 'prompt:0.5']),
        (openslr, 0.4, ['notimestamp', 'cluster:speech2', 'prompt:0.5']),
        (openCpop, 0.1, ['order', 'pad', 'cluster:opencpop', 'prompt:0.5']),
        (openCpop, 0.1, ['pad', 'cluster:opencpop2', 'prompt:0.5']),
        ])
    speech_collate_fn = SpeechDataCollatorWithPadding(auto_merge_tensor=False)
    opencpop_collate_fn = OpenCpopDataCollatorWithPadding(auto_merge_tensor=False)
    collate_fn = mixedDataCollator({
        'speech': speech_collate_fn,
        'speech2': speech_collate_fn,
        'opencpop': opencpop_collate_fn,
        'opencpop2': opencpop_collate_fn
    })
    print('wrapped_dataset', train, len(wrapped_dataset))
    dataloader = naive_dataloader(wrapped_dataset, collate_fn, train)
    return dataloader
    
train_dataloader = get_dataloader(True)
val_dataloader = get_dataloader(False)

# verify_dataloader_by_human(train_dataloader, model.tokenizer, SAMPLE_RATE)

'''
# Run learning rate finder
from pytorch_lightning.tuner import Tuner
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, 
                          train_dataloaders=train_dataloader, 
                          val_dataloaders=val_dataloader,
                          num_training=100,
                          max_lr=0.01)
print(lr_finder.results)
fig = lr_finder.plot(suggest=True)
fig.savefig('logs/lr.png')
new_lr = lr_finder.suggestion()
print('new_lr', new_lr)
exit()
'''

# 4. fit the model
trainer.fit(
    model, 
    train_dataloader, 
    val_dataloader, 
    # ckpt_path = ckpt_path
    )