from matplotlib import pyplot as plt
from scr.datasets.common_voice import AudioMnistDataset
from torch.utils.data import Dataset, DataLoader
from scr.collator.collator import Collator
import torch
from torch import nn
from scr.logger.logger import AverageMeter
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from itertools import islice
from scr.model.quartznet import Quartznet
import torch_optimizer as optim
from scr.decoder.beam_search import *
from scr.configs.config import test_batch_size, num_iter_in_epoch_test,main_path_to_dataset_common_voice, path_to_csv_test_dataset, model_weights
import wandb
def test(ds, model_weights):
    wandb.login(key='358c4114387c5c7ca207c32ba4343e7c86efc182')
    wandb.init(project='QUARTZNET_voice', entity='mishaya') # username in wandb
    config = wandb.config
    test_dataloader = DataLoader(
        ds, batch_size=test_batch_size,
        shuffle=True, collate_fn=Collator(),
        num_workers=2, pin_memory=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Quartznet(input_channels=64, vocab=34).to(DEVICE)
    model.load_state_dict(torch.load(model_weights))
    criterion = nn.CTCLoss(blank=0).to(DEVICE)  # balnk ^ == 0 index
    history = defaultdict(list)
    validation_loss_meter = AverageMeter()
    val_average_cer = AverageMeter()
    val_average_wer = AverageMeter()
    model.eval()
    for i, batch in islice(enumerate(tqdm(test_dataloader)), num_iter_in_epoch_test):
        x = batch['specs'].to(DEVICE)
        y = batch['labels'].to(DEVICE)
        x_lengths = batch['specs length'].to(DEVICE)
        y_lengths = batch['labels lengths'].to(DEVICE)
        with torch.no_grad():
            val_log_probs = F.log_softmax(model(x), dim=1)# N C T
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
    print("\navg cer overall =", sum(history['cer']) / num_iter_in_epoch_test)
    print("avg wer overall =", sum(history['wer']) / num_iter_in_epoch_test)

def print_hi():
    TEST_DS = AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_test_dataset)

    test(TEST_DS, model_weights)

if __name__ == '__main__':
    print_hi()

