from pypinyin import lazy_pinyin, Style
import torch
from pydub import  AudioSegment
import numpy as np
from ttts.utils import vc_utils

MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-05-08-58-40/model-87.pt',
}
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F
cond_audio = 'ttts/1.wav'

device = 'cuda:0'
text = "大家好，今天来点大家想看的东西。"
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
pinyin = ' '+pinyin+' '
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
text_length = torch.LongTensor([text_tokens.shape[1]]).to(device)

from ttts.utils.infer_utils import load_model
import torchaudio
import json
import torchaudio.functional as F
from ttts.utils.data_utils import spectrogram_torch,HParams
hps = HParams(**json.load(open('ttts/vqvae/config_v3.json')))
wav, sr = torchaudio.load(cond_audio)
if wav.shape[0] > 1:
    wav = wav[0].unsqueeze(0)
wav = wav.to(device)
wav32k = F.resample(wav, sr, 32000)
wav32k = wav32k[:,:int(hps.data.hop_length * (wav32k.shape[-1]//hps.data.hop_length))]
wav = torch.clamp(wav32k, min=-1.0, max=1.0)
wav_length = torch.LongTensor([wav.shape[1]])
spec = spectrogram_torch(wav, hps.data.filter_length,
                    hps.data.hop_length, hps.data.win_length, center=False).squeeze(0)
spec_length = torch.LongTensor([
    x//hps.data.hop_length for x in wav_length]).to(device)
vqvae = load_model('vqvae', MODELS['vqvae.pth'], 'ttts/vqvae/config_v3.json', device)
with torch.no_grad():
    wav_out = vqvae.infer(text_tokens, text_length, spec.unsqueeze(0), spec_length)
torchaudio.save('gen_.wav', wav_out.squeeze(0).cpu(), 32000)
