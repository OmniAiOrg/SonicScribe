from torch.utils.data import DataLoader
import torchaudio
from data_reader.openslr_33 import Openslr33
from data_reader.openslr_38 import Openslr38
from data_reader.openslr_47 import Openslr47
from data_reader.openslr_68 import Openslr68
from utils.general_dataloader import WhisperOfficialBatch, whisper_official_batch_to_shape
from utils.whisper_dataloader import SpeechDataCollatorWithPadding, WrappedDataset


def verify_dataloader_by_human(dataloader: DataLoader, tokenizer, sample_rate):
    for b in dataloader:
        b: WhisperOfficialBatch = b
        # print(b)
        print(whisper_official_batch_to_shape(b))
        for i in range(b.mel.shape[0]):
            print('Labl', i, tokenizer.decode(
                b.data_label[i].tolist(), stop_at=tokenizer.eot))
            print('Mel', i, b.mel[i].shape)
            torchaudio.save('logs/verify_human.wav', b.waveform[i], sample_rate)
            input("enter for next")

if __name__ == '__main__':
    collect_fn = SpeechDataCollatorWithPadding(model = 'tiny', num_workers = 1, num_concat=4)

    def naive_dataloader(dataset, train):
        return  DataLoader(dataset, 
                    batch_size=4,
                    shuffle=True,
                    collate_fn=collect_fn,
                    num_workers = 1,
                    pin_memory=False,
                    persistent_workers=True
                    )
    
    def get_dataloader(train=True) -> DataLoader:
        dataset_a = Openslr38(train=train, key_filter=['audio', 'hanzi'])
        dataset_b = Openslr33(train=train, key_filter=['audio', 'hanzi'])
        dataset_c = Openslr47(train=train, key_filter=['audio', 'hanzi'])
        dataset_d = Openslr68(train=train, key_filter=['audio', 'hanzi'])
        wrapped_dataset = WrappedDataset([
            (dataset_a, 1, ['notimestamp']),
            (dataset_b, 1, ['notimestamp']),
            (dataset_c, 1, ['notimestamp']),
            (dataset_d, 1, ['notimestamp']),
            # (dataset_a, 1, ['notimestamp','pad']),
            # (dataset_b, 1, ['notimestamp','pad']),
            # (dataset_c, 1, ['notimestamp','pad']),
            # (dataset_d, 1, ['notimestamp','pad']),
            ])
        print('wrapped_dataset', train, len(wrapped_dataset))
        dataloader = naive_dataloader(wrapped_dataset, train)
        return dataloader
    
    test = get_dataloader(False)
    
    verify_dataloader_by_human(test, collect_fn.tokenizer, 16000)