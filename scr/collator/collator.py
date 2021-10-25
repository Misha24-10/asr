from math import ceil
import torch
from scr.precessing.wav2spec import Featurizer
from math import ceil

class Collator:

    def __call__(self, batch):
        mel_featurizer = Featurizer()
        wavs, labels = zip(*batch)

        lengths_for_wavs = torch.LongTensor([wav.size(-1) for wav in wavs])
        lengths_for_lables = torch.LongTensor([label.size(-1) for label in labels])
        num_elements_in_batch = len(lengths_for_wavs)

        specs = [mel_featurizer.forward(i) for i in wavs]
        lengths_for_specs = torch.LongTensor([i.shape[-1] for i in specs])

        spec_tensors = torch.zeros(num_elements_in_batch, specs[0].shape[1], max(lengths_for_specs))
        for i, spec in enumerate(specs):
            spec_tensors[i] = torch.cat([spec,
                                         torch.zeros(1, spec.shape[1],
                                                     max(lengths_for_specs)- spec.shape[-1])],
                                        dim=2)


        batch_wavs = torch.zeros(len(batch), max(lengths_for_wavs),dtype=torch.float32)
        batch_label = torch.zeros(len(batch), max(lengths_for_lables), dtype=torch.long)

        for i, (wav, length) in enumerate(zip(wavs, lengths_for_wavs)):
            batch_wavs[i, :(wav.size(-1))] = wav.squeeze()

        for i, (string, length) in enumerate(zip(labels, lengths_for_lables)):
            batch_label[i, :length] = string


        lengths_for_specs = torch.tensor([ceil(x / 2) for x in lengths_for_specs])

        return {
            'wavs': batch_wavs,
            'wavs lengths': lengths_for_wavs,
            'labels': batch_label,
            'labels lengths': lengths_for_lables,
            'specs': spec_tensors,
            "specs length": lengths_for_specs
        }

from math import ceil

class Collator_transforms:

    def __call__(self, batch):
        mel_featurizer = Featurizer()
        wavs, labels = zip(*batch)

        lengths_for_wavs = torch.LongTensor([wav.size(-1) for wav in wavs])
        lengths_for_lables = torch.LongTensor([label.size(-1) for label in labels])
        num_elements_in_batch = len(lengths_for_wavs)

        specs = [mel_featurizer.forward(i,True) for i in wavs]
        lengths_for_specs = torch.LongTensor([i.shape[-1] for i in specs])

        spec_tensors = torch.zeros(num_elements_in_batch, specs[0].shape[1], max(lengths_for_specs))
        for i, spec in enumerate(specs):
            spec_tensors[i] = torch.cat([spec,
                                         torch.zeros(1, spec.shape[1],
                                                     max(lengths_for_specs)- spec.shape[-1])],
                                        dim=2)


        batch_wavs = torch.zeros(len(batch), max(lengths_for_wavs),dtype=torch.float32)
        batch_label = torch.zeros(len(batch), max(lengths_for_lables), dtype=torch.long)

        for i, (wav, length) in enumerate(zip(wavs, lengths_for_wavs)):
            batch_wavs[i, :(wav.size(-1))] = wav.squeeze()

        for i, (string, length) in enumerate(zip(labels, lengths_for_lables)):
            batch_label[i, :length] = string


        lengths_for_specs = torch.tensor([ceil(x / 2) for x in lengths_for_specs])

        return {
            'wavs': batch_wavs,
            'wavs lengths': lengths_for_wavs,
            'labels': batch_label,
            'labels lengths': lengths_for_lables,
            'specs': spec_tensors,
            "specs length": lengths_for_specs
        }