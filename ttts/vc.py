import torch
import numpy as np
from ttts.utils import vc_utils
from ttts.utils.infer_utils import load_model
import torchaudio
import json
import torchaudio.functional as F
from ttts.utils.data_utils import spectrogram_torch,HParams
device = 'cuda:0'
hps = HParams(**json.load(open('ttts/vqvae/config_v3.json')))
def wav2spec(wav_path):
    wav, sr = torchaudio.load(wav_path)
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
    return spec,spec_length
    

MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-15-13-44-46/model-92.pt',
}
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
cond_audio = 'ttts/3.wav'
in_audio = 'ttts/6.wav'
source, source_length = wav2spec(in_audio)
refer, refer_length = wav2spec(cond_audio)

vqvae = load_model('vqvae', MODELS['vqvae.pth'], 'ttts/vqvae/config_v3.json', device)
with torch.no_grad():
    wav_out = vqvae.vc(source.unsqueeze(0), source_length, refer.unsqueeze(0), refer_length)
torchaudio.save('vc.wav', wav_out.squeeze(0).cpu(), 32000)