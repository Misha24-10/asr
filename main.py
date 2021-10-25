from matplotlib import pyplot as plt
from scr.datasets.common_voice import AudioMnistDataset
from torch.utils.data import Dataset, DataLoader
from scr.collator.collator import Collator
import torch
from torch import nn
from scr.model.model import Jasper
from scr.precessing.wav2spec import Featurizer
from collections import defaultdict
from scr.logger.logger import AverageMeter, Timer
from tqdm import tqdm
from scr.metrics.metrics import calculate_cer
from collections import defaultdict
import torch.nn.functional as F
from itertools import islice
from scr.model.quartznet import Quartznet
import torch_optimizer as optim
from scr.decoder.beam_search import *


def test(ds, model_weights):
    test_dataloader = DataLoader(
        ds, batch_size=20,
        shuffle=True, collate_fn=Collator(),
        # num_workers=2,
        pin_memory=True)
    NUM_EPOCH = 20
    history = defaultdict(list)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Quartznet(input_channels=64, vocab=34).to(DEVICE)
    model.load_state_dict(torch.load(model_weights))
    criterion = nn.CTCLoss(blank=0).to(DEVICE)  # balnk ^ == 0 index
    optimizer = optim.NovoGrad(model.parameters(), lr=0.05,betas=(0.95, 0.5),weight_decay=0.001)
    history = defaultdict(list)
    validation_loss_meter = AverageMeter()
    val_average_cer = AverageMeter()
    val_average_wer = AverageMeter()
    model.eval()
    for i, batch in islice(enumerate(tqdm(test_dataloader)), NUM_EPOCH):
        x = batch['specs'].to(DEVICE)
        y = batch['labels'].to(DEVICE)
        x_lengths =  batch['specs length'].to(DEVICE)
        y_lengths =  batch['labels lengths'].to(DEVICE)
        with torch.no_grad():
            val_log_probs = F.log_softmax(model(x),dim=1)# N C T
            val_cer_beam, val_wer_beam, _ = calculate_cer_beam(y, beam_serch_eval(val_log_probs),y_lengths)
            print("\nVall cer (beam search)=", val_cer_beam)
            print("Vall wer (beam search) =", val_wer_beam)
            val_log_probs = val_log_probs.permute(2, 0, 1) # T N C
            val_loss = criterion(val_log_probs, y, x_lengths, y_lengths)
            argmax_decoding = val_log_probs.detach().cpu().argmax(dim=-1).transpose(0, 1)
            val_cer, val_wer, pairs  = calculate_cer(y, argmax_decoding, y_lengths, x_lengths)
            val_average_cer.update(val_cer)
            val_average_wer.update(val_wer)
            validation_loss_meter.update(val_loss.item())
            history['cer'].append(val_average_cer.avg)
            history['wer'].append(val_average_wer.avg)
        print("Vall cer (argmax)=", val_cer)
        print("Vall wer (argmax) =", val_wer)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flatten()
    axes[0].plot(history['cer'], label='CER')
    axes[0].legend(); axes[0].grid()

    axes[1].plot(history['wer'], label='WER')
    axes[1].legend(); axes[1].grid()
    print("avg cer=",sum(history['cer'])/NUM_EPOCH)
    print("avg wer=",sum(history['wer'])/NUM_EPOCH)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    Main_path = "/content/cv-corpus-7.0-2021-07-21/ru/clips/"
    path_to_csv_test = '/content/cv-corpus-7.0-2021-07-21/ru/test.tsv' # Just for get score
    model_weights = '/content/gdrive/MyDrive/AUDIO_DLA/asr_last_week/state_dict_model_commonvoice_part9_plus_aug_in_spec.pt'
    TEST_DS = AudioMnistDataset(Main_path, path_to_csv_test)

    test(TEST_DS,model_weights)

if __name__ == '__main__':
    print_hi('PyCharm')

