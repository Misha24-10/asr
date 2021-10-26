from torch import nn
from torchaudio.transforms import MelSpectrogram
import torch
import torchaudio
from torchaudio_augmentations import *
from matplotlib import pyplot as plt

transforms = [
    RandomApply([Noise(min_snr=0.05, max_snr=0.3)], p=0.1), # add noise in audio
    RandomApply([PolarityInversion()], p=0.1), #elements negative->positive, posetive->negative out=−1×input https://www.hackaudio.com/digital-signal-processing/amplitude/gain-change/polarity-inversion/
    RandomApply([Gain(min_gain=-20, max_gain=-1)], p=0.1), #decrease volume
    RandomApply([Gain(min_gain=-1, max_gain=25)], p=0.1), #increase volume
    RandomApply([Reverb(sample_rate=8000)], p=0.1) #echo like in room
]


def apply_compression(melspec):
    return torch.log(melspec.clamp(1e-5))

class Featurizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.featurizer = MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            power=2.0
        )

        self.featurizer_trans = nn.Sequential(
            self.featurizer,
            torchaudio.transforms.TimeMasking(time_mask_param=5),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=5))

    def forward(self, x, transforms=None):


        if transforms:
            x = [
                apply_compression(self.featurizer_trans(sample))
                for sample in x
            ]
        else:
            x = [
                apply_compression(self.featurizer(sample))
                for sample in x
            ]

        x = torch.stack(x, dim=0)
        return x

def visualize_audio(wav: torch.Tensor, sr: int = 22050):
    # Average all channels
    if wav.dim() == 2:
        # Any to mono audio convertion
        wav = wav.mean(dim=0)

    plt.figure(figsize=(20, 5))
    plt.plot(wav, alpha=.7, c='green')
    plt.grid()
    plt.xlabel('Time', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.show()