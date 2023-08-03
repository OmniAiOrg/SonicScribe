from model.sonic_scriber import SonicScriber
from utils.general_dataloader import WhisperDataCollatorWithPadding
from utils.load_checkpoint import get_whisper_checkpoint

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

if __name__ == '__main__':
    collate_fn=WhisperDataCollatorWithPadding('tiny')
    sonic_scriber = SonicScriber('tiny', collate_fn)
    initialize_sonic_scriber_from_whisper(sonic_scriber)