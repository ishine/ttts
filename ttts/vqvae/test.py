from ttts.vqvae.augment import Augment
import torchaudio.functional as AuF
import torchaudio
import json
from ttts.utils.data_utils import HParams
from typing import List, Optional, Tuple, Union
import torch
from ttts.utils.infer_utils import load_model
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from pypinyin import lazy_pinyin, Style
import torch.nn.functional as F
device = 'cuda:0'
vqvae_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-04-24-05-25-41/model-61.pt'
model = load_model('vqvae', vqvae_path, 'ttts/vqvae/config_v3.json', device)
audio1 = 'ttts/4.wav'
def get_audio(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav[0].unsqueeze(0)
    wav32k = AuF.resample(wav, sr, 32000)
    wav32k = wav32k[:,:int(320 * 4 * (wav32k.shape[-1]//320//4))]
    wav32k = torch.clamp(wav32k, min=-1.0, max=1.0)
    return wav32k
audio1 = get_audio(audio1)
spec1 = spectrogram_torch(audio1, 1024,
                        320, 1024, center=False).squeeze(0)
spec_length1 = torch.LongTensor([spec1.size(1)]).to(device)
text = '霞浦县衙城镇乌旗，瓦窑村水位猛涨。'
# text = '涨。'
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
text = text_tokens
text_length = torch.LongTensor([text.size(1)]).to(device)
spec1 = spec1.to(device)
audio = model.infer(spec1.unsqueeze(0),spec_length1,
                    text, text_length, spec1.unsqueeze(0),spec_length1)
torchaudio.save('tt1.wav',audio[0].cpu(),32000)