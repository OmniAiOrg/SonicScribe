import os
import torchaudio
from dataclasses import dataclass
from utils.whisper_duration_auto_tag import WhisperDurationTagger

@dataclass
class TransChar:
    word: str
    start: float
    end: float

def process_audio(input_path: str, output_path: str, trans_chars: list[TransChar]):
    # Check if output directory exists, create it if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load audio file using torchaudio
    waveform, sample_rate = torchaudio.load(input_path)

    # Resample to 16000 Hz
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    # Process each TransChar
    i = 1
    for trans_char in trans_chars:
        # Calculate start and end samples
        start_sample = int(trans_char.start * 16000)
        end_sample = int(trans_char.end * 16000)

        # Extract audio segment
        segment = waveform[:, start_sample:end_sample]

        # Save audio segment to file
        output_file = os.path.join(output_path, f"{i}-{trans_char.word}.wav")
        i += 1
        torchaudio.save(output_file, segment, 16000)
        
if __name__ == '__main__':
    # Define input and output paths
    def verify_whisper_tagger(input_path, text):
        output_path = './assets/sample_audio/duration_test/'+text
        tagger = WhisperDurationTagger('tiny')
        trans_chars = tagger.predict(text, input_path)
        for sc in trans_chars:
            print(f'{sc.word}: {sc.start} -> {sc.end}')
        # Process audio
        process_audio(input_path, output_path, trans_chars)
        
    input_path = './assets/sample_audio/33-Aishell/data_aishell/wav/dev/S0726/BAC009S0726W0171.wav'
    text = '该土地拍卖价刷新了杨浦区土地最贵单价记录'
    # input_path = './assets/sample_audio/opencpop/2003000102.wav'
    # text = '如果云层是天空的一封信'
    verify_whisper_tagger(input_path, text)
        
    def verify_opencpop_official_duration(index):
        from data_reader.opencpop import OpenCpop
        oc_test = OpenCpop(train=False)
        audio_dir, hanzi_words, note_duration, text = oc_test.get_naive_item(index)
        output_path = './assets/sample_audio/duration_test/'+text
        trans_chars = []
        note_duration_cum = 0
        for i in range(len(hanzi_words)):
            trans_chars.append(
                TransChar(
                    hanzi_words[i],
                    note_duration_cum,
                    note_duration_cum + note_duration[i]
                )
            )
            note_duration_cum += note_duration[i]
        process_audio(audio_dir, output_path, trans_chars)
    verify_opencpop_official_duration(0)
    '''
    As we can hear from the output audio, the opencpop dataset is the best.
    Whisper tagger always brings a little bit of next character into the current one.
    But I decide not to change it. I'll use different prompt for different tag method.
    Maybe <|whisper_tagger|> and <|human_tagger|>, this way is easier and may works well. 
    '''
        
    