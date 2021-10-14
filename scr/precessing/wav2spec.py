from torch import nn
from torchaudio.transforms import MelSpectrogram
import torch


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

    def forward(self, x, x_len=None):
        x = [
            apply_compression(self.featurizer(sample))
            for sample in x
        ]
        x = torch.stack(x, dim=0)
        return x
