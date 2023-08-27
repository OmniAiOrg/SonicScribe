from model.sonic_scriber import SonicScriber
from model.whisper_official import WhisperOfficial
from utils.general_dataloader import WhisperDataCollatorWithPadding, WhisperOfficialDataCollatorWithPadding
from utils.load_checkpoint import get_whisper_checkpoint
import torch.nn.functional as F

def initialize_sonic_scriber_from_whisper(model: SonicScriber):
    checkpoint = get_whisper_checkpoint(name = model.model_size)
    new_checkpoint = {}
    for k, v in checkpoint["model_state_dict"].items():
        if 'decoder.blocks.3' in k:
            new_checkpoint[f'{k.replace("blocks.3", "head_block.hanzi")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.pinyin")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.note")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.tone")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.slur")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.start")}'] = v
            new_checkpoint[f'{k.replace("blocks.3", "head_block.end")}'] = v
        else:
            new_checkpoint[f'{k}'] = v
    model.load_state_dict(new_checkpoint, strict = False)
    
def initialize_whisper_official_from_whisper(model: WhisperOfficial):
    checkpoint = get_whisper_checkpoint(name = model.model_size)
    new_checkpoint = {}
    for k, v in checkpoint["model_state_dict"].items():
        if 'token_embedding' in k:
            p = [p for n, p in model.named_parameters() if n==k][0]
            print(f'k={k}, p={p.shape}')
            assert p.shape[0] >= v.shape[0], f'p.shape={p.shape}, v.shape={ v.shape}'
            padded_v = F.pad(v, (0,0,0, p.shape[0] - v.shape[0]), mode='constant', value=0)
            print(f'v={v.shape} padded_v={padded_v.shape}')
            new_checkpoint[k] = padded_v
        else:
            new_checkpoint[f'{k}'] = v
    model.load_state_dict(new_checkpoint, strict = False)

if __name__ == '__main__':
    # collate_fn=WhisperDataCollatorWithPadding('tiny')
    # sonic_scriber = SonicScriber('tiny', collate_fn)
    # initialize_sonic_scriber_from_whisper(sonic_scriber)
    
    collate_fn=WhisperOfficialDataCollatorWithPadding('tiny')
    whisper_official = WhisperOfficial('tiny')
    initialize_whisper_official_from_whisper(whisper_official)
    
    