# TTTS_v2(WIP)

## V2 is built upon the VALL-E style GPT, and VQ-VAE-GAN is based on HierachySpeech++ and GPT-SoVITS.

# Demo
Coming soon.

# Install
```
pip install -e .
```
# Training
Training the model including two steps.

### 1. Tokenizer training
Use the `ttts/prepare/bpe_all_text_to_one_file.py` to merge all text you have collected. To train the tokenizer, check the `ttts/gpt/voice_tokenizer` for more info.

### 2. VQVAE training
Use the `1_vad_asr_save_to_jsonl.py` to preprocess dataset.
Use the following instruction to train the model.
```
python ttts/vqvae/train.py
```
There are three stages for vqvae training, and this is important for the vqgan to converge.
- stage1: Use data augmentation and continuous latent feature to train the timbre disentangle module. vq and vq2 in `config.json` are all set to false.
- stage2: Add vq to the model and freeze the timbre remove module. vq is set to True, vq2 is set to False.
- stage3: Freeze vq and timbre remover, train the decoder part only. vq and vq2 are all set to True.

### 3. GPT training
Use `2_save_vq_to_disk.py` to preprocess vq. 

It's important to concat long enough samples for training, so you need to check the `dataset.py` to make true the same speaker audio can be recognized.

Run
```
accelerate launch ttts/gpt/train.py
```
to train the model.

