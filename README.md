# SonicScribe

SonicScribe is a project that converts a singer's voice into a script with notes, phonemes, and duration.

TODO:

- [x] Create a dataset for openCpop.
- [x] Whisper auto duration tag for speech dataset
- [ ] ~~visulize dataset to verify all positions match~~
- [x] split audio to word level file and verify by listen to check the duration tag.
- [x] Create a dataset for openslr33.
- [x] Define an omni data format for training that can handle singing (phoneme, note, duration), music (note), and speech (phoneme, duration) together.
- [x] Create a tokenizer for single Chinese characters. The tokenizer should be dynamic during training and should only support simplified Chinese characters. For new single characters, start from the mean of their components.
- [x] Create a tokenizer for notes, duration, and phonemes. For phonemes, copy from existing or similar phonemes.
- [x] Create combined_dataset and dataloader which with different weights for different datasets
- [x] Create a loss function for a mixture of different kinds of datasets.
- [x] Create a data reader to mix different kinds of datasets.
- [ ] for different dataset, use different duration token embedding, such that opencpop can be trained better
- [x] Create a trainer that can train this modified whisper (SonicScribe).
- [x] Fix weighted dataset, always read from dataset instead of use dataloader snapshot
- [x] Log validation txt to tensorboard
- [ ] clean the dataset with remove silent before and after sentence
- [ ] predict hanzi -> partial with AP SP -> fintune in opencpop hanzi only -> add note -> add slur -> try calculate duration
- [ ] new training logic: finetune official whisper -> add a head to predict single token hanzi -> add note, slur ,... 
- [ ] new data format: do not use special tokenizer, instead, use official multi-token pbe. Then you can train the model with rearrange dataformat:
    - example: <|simple transcribe|> <|chinese|> 明天会更好 <|eot|>
    - example: <|transcribe|> <|chinese|> <|1|> 明 <|2|> 天 <|eot|>
    - example: <|simple music|> <|piano|> <|A4|> <|B3|> <|eot|>
    - example: <|music|> <|piano|> <|1|> <|A4|> <|2|> <|B3|> <|eot|>
    - example: <|singing|> <|male|> <|1|> 明 <|B3|> <|2|> 天 <|A4|> <|eot|>
    - in inference, <|1|> will not be predicted, instead, it will be hardcode as prompt
    - cuation! btw, remove all multi chinese with single token in tokenizer!
    - what about use this in english song? it may works! 
    - example: <|singing|> <|en|> <|male|> <|1|> fan <|B3|> <|2|> tas <|A4|> <|3|> tic <|A5|>  <|eot|>
- [ ] Modify the model with single embedding input which concats the embeddings, which could be faster and more accurate
- [ ] Create an inference model that can be easily used with command line and HTTP/web.
- [ ] Generate multiple types of export formats.
- [ ] Link this ASR with TTS like diffsinger.

> Note: There is no tiktok encoding in this project because we use a simple method to pair notes, characters, and phonemes.