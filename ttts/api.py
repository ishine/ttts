from pypinyin import lazy_pinyin, Style
import torch
from pydub import  AudioSegment
import numpy as np
from ttts.utils import vc_utils
import cutlet
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
import torchaudio
from pypinyin import pinyin, lazy_pinyin, Style

device = 'cuda:0'
cond_audio = 'ttts/6.wav'
lang = 'ZH'
text = "大家好，今天来点大家想看的东西。"
# cond_audio = 'ttts/jp.wav'
# lang = 'JP'
# text = '皆さん、こんにちは！今日は皆さんに読んでいただきたいことがあります。'
# cond_audio = 'ttts/en.wav'
# lang = 'EN'
# text = "Hello everyone, here's something you'll want to read today."
# cond_audio = 'ttts/kr.wav'
# lang = 'KR'
# text = "안녕하세요, 여러분, 오늘 읽어보시면 좋을 내용이 있습니다."

katsu = cutlet.Cutlet()
r = Transliter(academic)
def convert_to_latin(text, lang):
    if lang == "ZH":
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
    elif lang == "JP":
        text = katsu.romaji(text)
        text = ' '+text+' '
    elif lang == "EN":
        text = text
        text = ' '+text+' '
    elif lang == "KR":
        text = r.translit(text)
        text = ' '+text+' '
    else:
        return None
    return text

MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-31-16-58-34/model-555.pt',
}
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F

tok_zh = VoiceBpeTokenizer('ttts/tokenizers/zh_tokenizer.json')
tok_en = VoiceBpeTokenizer('ttts/tokenizers/en_tokenizer.json')
tok_jp = VoiceBpeTokenizer('ttts/tokenizers/jp_tokenizer.json')
tok_kr = VoiceBpeTokenizer('ttts/tokenizers/kr_tokenizer.json')
text = convert_to_latin(text,lang)
if lang == "ZH":
    text = tok_zh.encode(text.lower())
elif lang == "JP":
    text = tok_jp.encode(text.lower())
elif lang == "EN":
    text = tok_en.encode(text.lower())
elif lang == "KR":
    text = tok_kr.encode(text.lower())
else:
    print('not supprt lang')
text = torch.LongTensor(text).unsqueeze(0)
if lang == "ZH":
    text = text + 256*0
    lang = 0
elif lang == "JP":
    text = text + 256*1
    lang = 1
elif lang == "EN":
    text = text + 256*2
    lang = 2
elif lang == "KR":
    text = text + 256*3
    lang = 3

text_tokens = text.to(device)
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
torchaudio.save('gen.wav', wav_out.squeeze(0).cpu(), 32000)
