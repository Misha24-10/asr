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
import wandb

def train(ds):
    wandb.init(project='asr', entity='mishaya') # username in wandb
    config = wandb.config          # Initialize config
    config.batch_size = 20
    config.epochs = 50             # number of epochs to train (default: 10)
    config.lr = 3e-3               # learning rate (default: 0.01)
    config.log_interval = 1     # how many batches to wait before logging training status

    train_dataloader = DataLoader(
        ds, batch_size=20,
        shuffle=False, collate_fn=Collator(),
        num_workers=2,
        pin_memory=True)

    one_batch = next(iter(train_dataloader))
    NUM_EPOCH = 100
    history = defaultdict(list)
    DEVICE = torch.device('cuda')

    model = Jasper(input_channels=64, vocab=34).to(DEVICE)
    criterion = nn.CTCLoss(blank=32, )  # balnk ^ == 32 index
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
    mel_featurizer = Featurizer().to(DEVICE)
    average_cer = AverageMeter()
    average_wer = AverageMeter()
    average_loss = AverageMeter()
    wandb.watch(model, log_freq=1)
    for epoch in range(NUM_EPOCH):
        with Timer('epoch') as epoch_t:
            for batch in tqdm(range(NUM_EPOCH)):
                with Timer('iteration') as iteration_t:
                    x = one_batch['specs'].to(DEVICE)
                    y = one_batch['labels'].to(DEVICE)
                    x_lengths = one_batch['specs length'].to(DEVICE)
                    y_lengths = one_batch['labels lengths'].to(DEVICE)

                    log_probs = model(x)  # N C T
                    log_probs = log_probs.permute(2, 0, 1)  # T N C
                    loss = criterion(log_probs, y, x_lengths, y_lengths)

                history['time_per_iter'].append(iteration_t.t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                argmax_decoding = log_probs.detach().cpu() \
                    .argmax(dim=-1).transpose(0, 1)

                cer, wer, pairs = calculate_cer(y, argmax_decoding, y_lengths, x_lengths)
                average_cer.update(cer)
                average_wer.update(wer)
                average_loss.update(loss.item())
                history['loss'].append(average_loss.avg)
                history['cer'].append(average_cer.avg)
                history['wer'].append(average_wer.avg)
                print(cer)
                print(wer)
                wandb.log({
                    "loss": loss.item(),
                    "cer": cer,
                    "wer": wer})
            history['time_per_epoch'].append(epoch_t.t)

        print(f'Epoch: {epoch}')

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()
        axes[0].plot(history['loss'], label='Loss')
        axes[1].plot(history['cer'], label='CER')
        axes[0].legend();
        axes[0].grid()
        axes[1].legend();
        axes[1].grid()

        axes[2].plot(history['time_per_iter'], label='iter')
        axes[2].legend();
        axes[2].grid()
        axes[3].plot(history['wer'], label='WER')
        axes[3].legend();
        axes[3].grid()

        plt.show()


def work():
    # Use a breakpoint in the code line below to debug your script.
    Main_path = "/content/cv-corpus-7.0-2021-07-21/ru/clips/"
    path_to_csv = '/content/cv-corpus-7.0-2021-07-21/ru/train.tsv'
    ds = AudioMnistDataset(Main_path, path_to_csv)
    train(ds)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
