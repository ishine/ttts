import torchaudio
import torch
from torchaudio.functional import detect_pitch_frequency, pitch_shift
from torchaudio.functional import phase_vocoder, resample, spectrogram
from math import log10
from torch import nn
import time
device = 'cuda:6'
# Torch implement of pitch shift and formant shift
# https://github.com/drfeinberg/Parselmouth-Guides/blob/master/ManipulatingVoices.ipynb
def get_pitch(wav, sample_rate):
    pitch = detect_pitch_frequency(
        wav,
        sample_rate,
    )
    return pitch
class torch_augment(nn.Module):
    def __init__(self,
                 sample_rate,
                 formant_factor):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(power=None)
        self.inverspectrogram = torchaudio.transforms.InverseSpectrogram()
        self.stretch = torchaudio.transforms.TimeStretch()
        self.sample_rate = sample_rate
        self.formant_factor = formant_factor
    def sampler(self, bsize, ratio):
        shifts = torch.rand(bsize) * (ratio - 1.) + 1.
        # flip
        flip = torch.rand(bsize) < 0.5
        shifts[flip] = shifts[flip] ** -1
        return shifts
    def formant_shift(self, wav,  factor=1.4):
        sample_rate = self.sample_rate
        duration = wav.shape[-1]/sample_rate
        resampled_wav = resample(
            wav,
            sample_rate,
            int(sample_rate*factor)
        )
        spec = self.spectrogram(resampled_wav)
        resampled_spec = self.stretch(spec, float(factor))
        resampled_wav = self.inverspectrogram(resampled_spec,wav.shape[-1])
        pitch = get_pitch(wav,sample_rate)
        resampled_pitch = get_pitch(resampled_wav,sample_rate)
        formant_factor = torch.mean(pitch)/torch.mean(resampled_pitch)
        out = pitch_shift(
            resampled_wav,
            sample_rate,
            log10(formant_factor) / log10(2),
        )
        return out
    def forward(self,wav):
        factor = self.sampler(wav.shape[0], self.formant_factor).to(wav.device)
        print(factor)
        return self.formant_shift(wav, factor)

def pitch_shift_(wav, sample_rate, step):
    wav_shift = pitch_shift(wav,sample_rate, step)
    return wav_shift
wav, sr = torchaudio.load('ttts/0.wav')
# augment = torch_augment(sr,1.5).to(device)
wav = wav.to(device)
for i in [-16,-15,-14,-13,-11,-9,-8,-7,-4,3,4,5,6,7,8,9,10,12,13]:
    print(i)
    t = time.time()
    wav_aug = pitch_shift_(wav,sr,i).cpu()
    print(time.time()-t)
# torchaudio.save('ttts/0_aug.wav', wav_aug, sr)