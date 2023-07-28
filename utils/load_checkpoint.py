import whisper
import yaml
import os
import torch

SETTINGS_YAML = './utils/settings.yaml'

def get_config(config_file = SETTINGS_YAML):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config
    
def get_whisper_checkpoint(name='tiny', config_file = SETTINGS_YAML):
    config = get_config(config_file)
    checkpoint_path = config['whisper']['checkpoint']
    checkpoint_file = checkpoint_path + f'/{name}.pt'
    with open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location='cpu')
        return checkpoint
    
def get_whisper_token_embedding(name='tiny', config_file = SETTINGS_YAML):
    checkpoint = get_whisper_checkpoint(name, config_file)
    token_embedding = checkpoint['model_state_dict']['decoder.token_embedding.weight']
    assert token_embedding.shape[0] > 50000 and token_embedding.shape[1] > 300, token_embedding.shape
    return token_embedding


        
if __name__ == "__main__":
    token_embedding = get_whisper_token_embedding()
    print(token_embedding.shape)