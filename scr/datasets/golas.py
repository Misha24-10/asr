from torch.utils.data import Dataset
from scr.decoder.char_text_encoder import Alphabet
import torchaudio
import torch
import pandas as pd
from torchaudio_augmentations import *
SR = 16_000
class AudioMnistDataset_Golas(Dataset):
    def __init__(self, path, path_to_data,transforms = None):
        DF = pd.read_json(path_to_data, lines=True)
        self.down_quantile = DF.quantile([.05, .95])['duration'].iloc[0]
        self.up_quantile = DF.quantile([.05, .95])['duration'].iloc[1]
        print("95% квантиль", self.up_quantile)
        print("5% квантиль", self.down_quantile)
        DF = DF[DF['duration'] <= self.up_quantile]
        DF = DF[DF['duration'] >= self.down_quantile][['audio_filepath', 'text']]
        self.table = DF
        self.mainpath = path
        self.paths = list(DF.audio_filepath)
        self.sentence = list(DF.text)
        self.transforms = transforms
    def __getitem__(self, index: int):


        path_to_wav = self.paths[index]
        label =  self.sentence[index]
        alph = Alphabet()
        label = torch.LongTensor(alph.char2int(label))

        wave_form, sample_rate = torchaudio.load(self.mainpath + path_to_wav)

        effects = [['gain', '-n'],['channels', '1'], ['rate', '16000']]
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(
            wave_form, sample_rate, effects, channels_first=True)

        if self.transforms:
            transform = Compose(transforms=self.transforms)
            wav = transform(wav)

        self.wav = wav
        self.sr =  sr
        return wav, label

    def __len__(self):
        return len(self.paths)