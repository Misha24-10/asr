from torch.utils.data import Dataset
from scr.decoder.char_text_encoder import Alphabet
import torchaudio
import torch
import pandas as pd
SR = 16_000


class AudioMnistDataset(Dataset):
    def __init__(self, path, path_to_data):
        DF = pd.read_csv(path_to_data, sep='\t')
        DF = DF[['path', 'sentence']]
        DF.dropna(subset=['path'], inplace=True)
        self.table = DF
        self.mainpath = path
        self.paths = list(DF.path)
        self.sentence = list(DF.sentence)

    def __getitem__(self, index: int):
        path_to_wav = self.paths[index]
        label = self.sentence[index]
        alph = Alphabet()
        label = torch.LongTensor(alph.char2int(label))

        wave_form, sample_rate = torchaudio.load(self.mainpath + path_to_wav)

        effects = [['gain', '-n'], ['channels', '1'], ['rate', '16000']]
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(
            wave_form, sample_rate, effects, channels_first=True)
        self.wav = wav
        self.sr = sr
        return wav, label

    def __len__(self):
        return len(self.paths)
