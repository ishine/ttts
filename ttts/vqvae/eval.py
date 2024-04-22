from ttts.utils import vc_utils as utils
import torchaudio
import torchaudio.functional as F
import json
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from ttts.vqvae.vq3 import SynthesizerTrn
from pypinyin import Style, lazy_pinyin
from ttts.utils.infer_utils import load_model
import torch
MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-04-10-07-11-14/model-73.pt',
}
hps_path='ttts/vqvae/config_v3.json'
hps = HParams(**json.load(open(hps_path)))
device = 'cuda:0'
vqvae = load_model('vqvae', MODELS['vqvae.pth'], 'ttts/vqvae/config_v3.json', device)
wav,sr = torchaudio.load('ttts/6.wav')
if wav.shape[0] > 1:
    wav = wav[0].unsqueeze(0)
wav_refer,sr_refer = torchaudio.load('ttts/5.wav')
if wav_refer.shape[0] > 1:
    wav_refer = wav_refer[0].unsqueeze(0)

wav32k = F.resample(wav, sr, 32000).to(device)
wav32k = wav32k[:,:int(hps.data.hop_length * 2 * (wav32k.shape[-1]//hps.data.hop_length//2))]
wav32k = torch.clamp(wav32k, min=-1.0, max=1.0)
wav_length = torch.LongTensor([wav32k.shape[-1]]).to(device)
spec = spectrogram_torch(wav32k, hps.data.filter_length,
                hps.data.hop_length, hps.data.win_length, center=False).squeeze(0)
spec_length = torch.LongTensor([
                x//hps.data.hop_length for x in wav_length]).to(device)

refer32k = F.resample(wav_refer, sr_refer, 32000).to(device)
refer32k = refer32k[:,:int(hps.data.hop_length * 2 * (refer32k.shape[-1]//hps.data.hop_length//2))]
refer32k = torch.clamp(refer32k, min=-1.0, max=1.0)
refer_length = torch.LongTensor([refer32k.shape[-1]]).to(device)
refer_spec = spectrogram_torch(refer32k, hps.data.filter_length,
                hps.data.hop_length, hps.data.win_length, center=False).squeeze(0)
refer_length = torch.LongTensor([
                x//hps.data.hop_length for x in refer_length]).to(device)

with torch.no_grad():
    wav_recon = vqvae.vc(spec.unsqueeze(0), spec_length,
        refer_spec.unsqueeze(0), refer_length, noise_scale=0.4)
torchaudio.save('gen.wav', wav_recon[0].detach().cpu(), 32000)