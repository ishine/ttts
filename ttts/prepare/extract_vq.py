import torchaudio
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.infer_utils import load_model
import torch
import torchaudio.functional as F
from tqdm import tqdm
import os
import json
from ttts.utils.data_utils import spectrogram_torch,HParams
import random

model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v2/G_128163.pth'
# device_ids = [4,5,6,7]
# rank = random.choice(device_ids)
# device = f'cuda:{rank}'
device = 'cuda'
vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config.json', device)
hps = HParams(**json.load(open('ttts/vqvae/config.json')))
def process_vq(path):
    wav_path = path
    try:
        wav,sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = wav.to(device)
        wav32k = F.resample(wav, sr, 32000)
        wav32k = wav32k[:,:int(hps.data.hop_length * 2 * (wav32k.shape[-1]//hps.data.hop_length//2))]
        wav = torch.clamp(wav32k, min=-1.0, max=1.0)
    except Exception as e:
        print(path)
        print(e)
        return
    try:
        with torch.no_grad():
            spec = spectrogram_torch(wav, hps.data.filter_length,
                    hps.data.hop_length, hps.data.win_length, center=False)
            code = vqvae.extract_code(wav, spec).squeeze(0).squeeze(0)
    except Exception as e:
        print(path)
        print(e)
        return
    outp = path+'.vq.pth'
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.save(code.tolist(), outp)
    return