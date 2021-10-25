from torch import nn


class Quartznet_Block(nn.Module):
    def __init__(self, input_ch, output_ch, kern_size):
        super().__init__()

        self.blocks = nn.ModuleList([ nn.ModuleList([
            nn.Conv1d(in_channels = input_ch if i==0 else output_ch,
                      out_channels=output_ch,
                      kernel_size=kern_size,
                      groups=input_ch if i==0 else output_ch,
                      padding= kern_size //2 ),
            nn.Conv1d(output_ch,
                      output_ch,
                      kernel_size=1),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(),
        ]) for i in range(5)
        ])

        self.residual_conv = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, kernel_size=1),
            nn.BatchNorm1d(output_ch),
        )
    def forward(self, x):
        x_befor_blocks = x
        for i, block_iter in enumerate(self.blocks):
            for j, layer in enumerate(block_iter):
                if  (i == len(self.blocks) - 1 and j == len(block_iter) - 1):
                    x = layer(self.residual_conv(x_befor_blocks) + x)
                else:
                    x = layer(x)
        return x

class Quartznet(nn.Module):
    def __init__(self, input_channels, vocab=34):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256,  kernel_size=33, stride=2, padding=33 // 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            Quartznet_Block(256,256,33),
            Quartznet_Block(256,256,39),
            Quartznet_Block(256,512,51),
            Quartznet_Block(512,512,63),
            Quartznet_Block(512,512,75)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, groups=512,  kernel_size=87, dilation=2, padding=86 ),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024,  kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=vocab,  kernel_size=1,  dilation=1),
        )

    def forward(self, x):
        e0 = self.conv1(x)
        BB = self.block1(e0)
        e1 = self.conv2(BB)
        e2 = self.conv3(e1)
        e3 = self.conv4(e2)
        return e3
