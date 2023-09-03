set -x
# python3 inference/transcribe.py /mnt/private_cq/dataset/opencpop/wavs/2050001872.wav --language Chinese --word_timestamps True

# audio="/mnt/private_cq/dataset/opencpop/wavs/2050001872.wav"
audio='assets/sample_audio/opencpop/2003000102.wav'
device=cuda
# audio="/mnt/private_cq/dataset/Corpus/zipped_file/ft_local/68-SLR68-MagicData/dev/37_5622/37_5622_20170913203118.wav"
whisper $audio --language Chinese --word_timestamps "True" --device $device --suppress_tokens "-1" -o "logs/test/" --condition_on_previous_text False
# python3 inference/transcribe.py $audio --language Chinese --word_timestamps True --device $device --suppress_tokens "" -o "logs/test/" --condition_on_previous_text False