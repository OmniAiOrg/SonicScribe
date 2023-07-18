# SonicScribe

SonicScribe is a project that converts a singer's voice into a script with notes, phonemes, and duration.

TODO:

- [x] Create a dataset for openCpop.
- [ ] Create a dataset for openslr.
- [ ] visulize dataset to verify all positions match 
- [ ] Define an omni data format for training that can handle singing (phoneme, note, duration), music (note), and speech (phoneme, duration) together.
- [ ] Create a tokenizer for single Chinese characters. The tokenizer should be dynamic during training and should only support simplified Chinese characters. For new single characters, start from the mean of their components.
- [ ] Create a tokenizer for notes, duration, and phonemes. For phonemes, copy from existing or similar phonemes.
- [ ] Create a loss function for a mixture of different kinds of datasets.
- [ ] Create a data reader to mix different kinds of datasets.
- [ ] Create a trainer that can train this modified whisper (SonicScribe).
- [ ] Create an inference model that can be easily used with command line and HTTP/web.
- [ ] Generate multiple types of export formats.
- [ ] Link this ASR with TTS like diffsinger.

> Note: There is no tiktok encoding in this project because we use a simple method to pair notes, characters, and phonemes.