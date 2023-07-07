# https://www.openslr.org/33

from data_reader.base_reader import BaseReader
import torch
from pathlib import Path
import os
import numpy as np
from torch import nn
import whisper
import torchaudio
import torchaudio.transforms as at
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple

class Openslr33(BaseReader):
    def __init__(self, train=True) -> None:
        super().__init__(train)
        # prepare the dataset and save to pickle for next time to use
        pickle_path = self.config['path']+''
        self.audio_transcript_pair_list
        
        
    def __getitem__(self, idx):
        pair = super().__getitem__(idx)
        
    
    

if __name__ == '__main__':
    openslr_33 = Openslr33()
    