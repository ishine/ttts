from pypinyin import lazy_pinyin, Style
import torch
from pydub import  AudioSegment
import numpy as np
from ttts.utils import vc_utils

MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-01-12-35-35/model-121.pt',
    'gpt.pth': '/home/hyc/tortoise_plus_zh/ttts/gpt/logs/2024-05-02-02-04-29/model-66.pt',
}
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F
cond_audio = 'ttts/0.wav'
cond_text = "霞浦县衙城镇乌旗，瓦窑村水位猛涨。"
# cond_text = "现场都是人，五辆警车好不容易找到位置停下。"
# cond_text = "嘿，现在是小事儿，只是向你打听一个地方。"
# cond_text = "除了曾经让全人类都畏惧的麻疹和天花之外。"
# cond_text = "开始步行导航，今天我也是没有迟到哦。"
# cond_text = "这是县交警队的一个小据点。"
# cond_text = "没什么没什么，只是平时他总是站在这里，有点奇怪而已。"
# cond_text = "没错没错，就是这样。"

device = 'cuda:0'
# text = "没错没错，就是这样。"
# text = "没什么没什么，只是平时他总是站在这里，有点奇怪而已。"
text = "大家好，今天来点大家想看的东西。"
# text = "霞浦县衙城镇乌旗，瓦窑村水位猛涨。"
# text = '高德官方网站，拥有全面、精准的地点信息，公交驾车路线规划，特色语音导航，商家团购、优惠信息。'
# text = '四是四，十是十，十四是十四，四十是四十。'
# text = '八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮。'
# text = '黑化肥发灰，灰化肥发黑。黑化肥挥发会发灰，灰化肥挥发会发黑。'
# text = '先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。诚宜开张圣听，以光先帝遗德，恢弘志士之气，不宜妄自菲薄，引喻失义，以塞忠谏之路也。'
text = cond_text + text
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
pinyin = ' '+pinyin+' '
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)

cond_pinyin = ' '.join(lazy_pinyin(cond_text, style=Style.TONE3, neutral_tone_with_five=True))
cond_pinyin = ' '+cond_pinyin+' '
cond_text_tokens = torch.IntTensor(tokenizer.encode(cond_pinyin)).unsqueeze(0).to(device)
cond_text_tokens = F.pad(cond_text_tokens, (0, 1))  # This may not be necessary.
cond_text_tokens = cond_text_tokens.to(device)
print(pinyin)
print(text_tokens)
from ttts.utils.infer_utils import load_model
import torchaudio
import json
import torchaudio.functional as F
from ttts.utils.data_utils import spectrogram_torch,HParams
# device = 'gpu:0'
gpt = load_model('gpt',MODELS['gpt.pth'],'ttts/gpt/config.json',device)
gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
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
cond_text_lengths = torch.LongTensor([cond_text_tokens.shape[1]]).to(device)
with torch.no_grad():
    cond_dur = vqvae.extract_dur(cond_text_tokens, cond_text_lengths, spec.unsqueeze(0), spec_length).squeeze(0)
print(cond_dur)
settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                     'top_p': .8, 'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
top_p = .8
temperature = .8
autoregressive_batch_size = 1
length_penalty = 2.0
repetition_penalty = 2.0
max_mel_tokens = 1000
print(text_tokens)
with torch.no_grad():
    dur = gpt.inference_speech(text_tokens,
                                    cond_dur,
                                    do_sample=True,
                                    top_p=top_p,
                                    temperature=temperature,
                                    num_return_sequences=autoregressive_batch_size,
                                    length_penalty=length_penalty,
                                    repetition_penalty=repetition_penalty,
                                    max_generate_length=max_mel_tokens)

print(dur)
codes = codes[:,:-1]
with torch.no_grad():
    tokens = text_tokens[:,cond_text_tokens.shape[1]:]
    lengths = torch.LongTensor([text_tokens.shape[1]-cond_text_tokens.shape[1]])
    out_wav = vqvae.decode(tokens, lengths, spec, spec_lengths, dur, noise_scale=0.5)
torchaudio.save('gen_.wav', out_wav.squeeze(0).cpu(), 32000)
