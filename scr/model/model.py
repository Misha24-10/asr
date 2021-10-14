from torch import nn


class Jasper_Block(nn.Module):
    def __init__(self, input_ch, output_ch, kernal_size, drop):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(in_channels=input_ch if i == 0 else output_ch, out_channels=output_ch, kernel_size=kernal_size,
                      padding=kernal_size // 2),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(),
            nn.Dropout(drop),
        ]) for i in range(5)
        ])

        self.residual_conv = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, kernel_size=1),
            nn.BatchNorm1d(output_ch),
        )

    def forward(self, x):
        x_befor_blocks = x
        for i, block in enumerate(self.blocks):
            for j, layer in enumerate(block):
                if (i == len(self.blocks) - 1 and j == len(block) - 2):
                    x = self.residual_conv(x_befor_blocks) + x
                    x = layer(x)
                else:
                    x = layer(x)
        return x


class Jasper(nn.Module):
    def __init__(self, input_channels, vocab=34):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=11, stride=2, padding=11 // 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.block1 = nn.Sequential(
            Jasper_Block(256, 256, 11, 0.2),
            Jasper_Block(256, 384, 13, 0.2),
            Jasper_Block(384, 512, 17, 0.2),
            Jasper_Block(512, 640, 21, 0.3),
            Jasper_Block(640, 768, 25, 0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=896, kernel_size=29, dilation=2, padding=28),
            nn.BatchNorm1d(896),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=896, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=vocab, kernel_size=1),
            nn.BatchNorm1d(vocab),
            nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        e0 = self.conv1(x)

        BB = self.block1(e0)

        e1 = self.conv2(BB)
        e2 = self.conv3(e1)
        e3 = self.conv4(e2)

        return e3.log_softmax(dim=1)
