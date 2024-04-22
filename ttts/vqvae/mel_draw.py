from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
import torchaudio.functional as AuF
from ttts.utils.utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
import torchaudio
import torch
import torchvision
from PIL import Image
y,sr = torchaudio.load('ttts/0.wav')
y = AuF.resample(y,sr,32000)

y_mel = mel_spectrogram_torch(
                    y.squeeze(1),
                    1024,
                    128,
                    32000,
                    640,
                    1024,
                    0,
                    None,
            )

img = plot_spectrogram_to_numpy(y_mel[0, :, :].detach().unsqueeze(-1).cpu())
img = Image.fromarray(img)
img.save('mel_20ms.png')
# torchvision.utils.save_image(torch.from_numpy(img), 'mel.png')