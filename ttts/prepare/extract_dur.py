import torchaudio
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.infer_utils import load_model
import torch
from pypinyin import Style, lazy_pinyin
import torchaudio.functional as F
from tqdm import tqdm
import os
import json
from ttts.utils.data_utils import spectrogram_torch,HParams
import random
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer

model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-01-11-57-50/model-121.pt'
device = 'cuda'
vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config_v3.json', device)
hps = HParams(**json.load(open('ttts/vqvae/config_v3.json')))
tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
def process_dur(wav_path, text):
    text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
    text = ' '+text+' '
    text = tok.encode(text)
    text = torch.LongTensor(text).to(device)
    text_length = torch.LongTensor([text.shape[0]]).to(device)

    try:
        wav,sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = wav.to(device)
        wav32k = F.resample(wav, sr, 32000)
        wav32k = wav32k[:,:int(hps.data.hop_length * (wav32k.shape[-1]//hps.data.hop_length))]
        wav = torch.clamp(wav32k, min=-1.0, max=1.0)
        wav_length = torch.LongTensor([wav.shape[1]])
    except Exception as e:
        print(wav_path)
        print(e)
        return
    try:
        with torch.no_grad():
            spec = spectrogram_torch(wav, hps.data.filter_length,
                hps.data.hop_length, hps.data.win_length, center=False).squeeze(0).to(device)
            spec_length = torch.LongTensor([
                x//hps.data.hop_length for x in wav_length]).to(device)
            dur = vqvae.extract_dur(text.unsqueeze(0),text_length, spec.unsqueeze(0), spec_length).squeeze(0).squeeze(0)
    except Exception as e:
        print(wav_path)
        print(e)
        return
    outp = wav_path+'.dur.pth'
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.save(dur.tolist(), outp)
    return