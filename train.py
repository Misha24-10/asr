from matplotlib import pyplot as plt
from scr.datasets.common_voice import AudioMnistDataset
from torch.utils.data import Dataset, DataLoader, Subset
from scr.collator.collator import Collator
import torch
from torch import nn
from scr.logger.logger import AverageMeter, Timer
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from itertools import islice
from scr.model.quartznet import Quartznet
import torch_optimizer as optim
from scr.decoder.beam_search import *
from scr.configs.config import num_iter_in_epoch_train,num_iter_in_epoch_valid,norm_clap, opt_learning_rate, opt_betas,opt_weight_decay ,main_path_to_dataset_common_voice, path_to_csv_test_dataset, model_weights, path_to_csv_train_dataset, path_to_csv_valid_dataset, train_batch_size, valid_batch_size,number_epoch
from scr.precessing.wav2spec import transforms
from scr.collator.collator import Collator_transforms
import wandb
def train(train_dataset, valid_dataset, model_weights = None):
    wandb.login(key='358c4114387c5c7ca207c32ba4343e7c86efc182')
    wandb.init(project='QUARTZNET_voice', entity='mishaya') # username in wandb
    config = wandb.config          # Initialize config
    config.batch_size = train_batch_size
    config.epochs = number_epoch             # number of epochs to train (default: 10)
    config.lr = opt_learning_rate               # learning rate (default: 0.01)
    config.opt_betas = opt_betas
    config.opt_weight_decay = opt_weight_decay
    config.log_interval = 1     # how many batches to wait before logging training status

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, collate_fn=Collator_transforms(),
        num_workers=2, pin_memory=True, drop_last=True
    )
    validation_dataloader = DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=True,
    collate_fn=Collator(),
    num_workers=2, pin_memory=False, drop_last=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Quartznet(input_channels=64, vocab=34).to(DEVICE)
    if model_weights:
        model.load_state_dict(torch.load(model_weights))
    criterion = nn.CTCLoss(blank=0).to(DEVICE)  # balnk ^ == 0 index
    optimizer = optim.NovoGrad(model.parameters(), lr=opt_learning_rate,betas=opt_betas,weight_decay=opt_weight_decay)
    history = defaultdict(list)
    for epoch in range(number_epoch):
        average_cer = AverageMeter()
        average_wer = AverageMeter()
        average_loss = AverageMeter()
        model.train()
        with Timer('epoch') as epoch_t:
            for i, batch in islice(enumerate(tqdm(train_dataloader)), num_iter_in_epoch_train):
                with Timer('iteration') as iteration_t:
                    model.float()  # add this here
                    x = batch['specs'].to(DEVICE)
                    y = batch['labels'].to(DEVICE)
                    x_lengths =  batch['specs length'].to(DEVICE)
                    y_lengths =  batch['labels lengths'].to(DEVICE)

                    log_probs =  F.log_softmax(model(x),dim=1) # N C T
                    log_probs = log_probs.permute(2, 0, 1) # T N C
                    loss = criterion(log_probs, y, x_lengths, y_lengths)
                history['time_per_iter'].append(iteration_t.t)
                optimizer.zero_grad()
                loss.backward()
                model.float() # add this here

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clap)
                if bool(torch.isnan(norm)):
                    print("\n","*"*150)
                    continue
                optimizer.step()

                argmax_decoding = log_probs.detach().cpu().argmax(dim=-1).transpose(0, 1)

                train_cer, train_wer, pairs  = calculate_cer(y, argmax_decoding, y_lengths, x_lengths)
                average_cer.update(train_cer)
                average_wer.update(train_wer)
                average_loss.update(loss.item())
                history['loss'].append(average_loss.avg)
                history['cer'].append(average_cer.avg)
                history['wer'].append(average_wer.avg)
                print('Train cer=', train_cer)
                print('Train wer', train_wer)
                wandb.log({
                    "loss": loss.item(),
                    "cer": train_cer,
                    "wer": train_wer},)
                if ( i  % 50 == 0 ):
                    wandb.log({
                        "Audio train every n step": [wandb.Audio(batch['wavs'][0], caption="Audio with aug", sample_rate=16000)],
                        "Spec train ": [wandb.Image(plt.imshow(batch['specs'][0]), caption="Spec with aug")]
                    })

            history['time_per_epoch'].append(epoch_t.t)
        print('--'*100)


    validation_loss_meter = AverageMeter()
    val_average_cer = AverageMeter()
    val_average_wer = AverageMeter()
    model.eval()
    for i, batch in islice(enumerate(tqdm(validation_dataloader)), num_iter_in_epoch_valid):
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
        print("Vall cer (argmax)=", val_cer)
        print("Vall wer (argmax) =", val_wer)

        wandb.log({
            "val_loss": val_loss.item(),
            "val_cer": val_cer,
            "val_wer": val_wer})
        if ( i  % 10 == 0  ):
            wandb.log({
                "Audio valid every n step": [wandb.Audio(batch['wavs'][0], caption="Audio with aug", sample_rate=16000)],
                "Spec valid ": [wandb.Image(plt.imshow(batch['specs'][0]), caption="Spec with aug")]
            })
    print(f'Epoch: {epoch}')


    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()
    axes[0].plot(history['loss'], label='Loss')
    axes[1].plot(history['cer'], label='CER')
    axes[0].legend(); axes[0].grid()
    axes[1].legend(); axes[1].grid()

    axes[2].plot(history['time_per_iter'], label='iter')
    axes[2].legend(); axes[2].grid()
    axes[3].plot(history['wer'], label='WER')
    axes[3].legend(); axes[3].grid()

    plt.show()

def print_hi():
    TEST_DS = AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_test_dataset)
    ds_train1 = AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_train_dataset, transforms)
    ds_train2 = AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_valid_dataset) # because valid dataset in large, I decided to train in part(90%) of validation DS
    train_dataset_1 = ds_train1
    train_ratio = 0.9
    train_size = int(len(ds_train2) * train_ratio)
    valtidation_size = len(ds_train2) -train_size
    indexes = torch.randperm(len(ds_train2))
    train_indexes = indexes[:train_size]
    validation_indexes = indexes[train_size:]

    train_dataset_2 = Subset(AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_valid_dataset, transforms), train_indexes)
    validation_dataset = Subset(AudioMnistDataset(main_path_to_dataset_common_voice, path_to_csv_valid_dataset), validation_indexes)
    print(len(train_dataset_1), len(train_dataset_2), len(validation_dataset))
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])


    train(train_dataset,validation_dataset, model_weights)

if __name__ == '__main__':
    print_hi()

