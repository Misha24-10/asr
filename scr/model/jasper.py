from torch import nn

def conv_block(in_f, out_f, dropout, *args, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_f, out_channels=out_f, *args, **kwargs,),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

def block_end(dropout):
    return nn.Sequential(
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Jasper_naive(nn.Module):
    def __init__(self, input_channels, vocab=34):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256,  kernel_size=11, stride=2, padding=11 // 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


        self.block1 = nn.Sequential(
            conv_block(256, 256, 0.2, kernel_size=11, padding=11 // 2),
            conv_block(256, 256, 0.2, kernel_size=11, padding=11 // 2),
            conv_block(256, 256, 0.2, kernel_size=11, padding=11 // 2),
            conv_block(256, 256, 0.2, kernel_size=11, padding=11 // 2),
            conv_block(256, 256, 0.2, kernel_size=11, padding=11 // 2)
        )
        self.one_to_one_conv_1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256)
        )
        self.block1_continues = block_end(0.2)


        self.block2 = nn.Sequential(
            conv_block(256, 384, 0.2, kernel_size=13, padding=13 // 2 ),
            conv_block(384, 384, 0.2, kernel_size=13, padding=13 // 2 ),
            conv_block(384, 384, 0.2, kernel_size=13, padding=13 // 2 ),
            conv_block(384, 384, 0.2, kernel_size=13, padding=13 // 2 ),
            conv_block(384, 384, 0.2, kernel_size=13, padding=13 // 2 ),
        )
        self.one_to_one_conv_2 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=1),
            nn.BatchNorm1d(384)
        )
        self.block2_continues = block_end(0.2)


        self.block3 = nn.Sequential(
            conv_block(384, 512, 0.2, kernel_size=17, padding=17 // 2 ),
            conv_block(512, 512, 0.2, kernel_size=17, padding=17 // 2 ),
            conv_block(512, 512, 0.2, kernel_size=17, padding=17 // 2 ),
            conv_block(512, 512, 0.2, kernel_size=17, padding=17 // 2 ),
            conv_block(512, 512, 0.2, kernel_size=17, padding=17 // 2 )
        )
        self.one_to_one_conv_3 = nn.Sequential(
            nn.Conv1d(384, 512, kernel_size=1),
            nn.BatchNorm1d(512)
        )
        self.block3_continues = block_end(0.2)


        self.block4 = nn.Sequential(
            conv_block(512, 640, 0.3, kernel_size=21, padding=21 // 2 ),
            conv_block(640, 640, 0.3, kernel_size=21, padding=21 // 2 ),
            conv_block(640, 640, 0.3, kernel_size=21, padding=21 // 2 ),
            conv_block(640, 640, 0.3, kernel_size=21, padding=21 // 2 ),
            conv_block(640, 640, 0.3, kernel_size=21, padding=21 // 2 )
        )
        self.one_to_one_conv_4 = nn.Sequential(
            nn.Conv1d(512, 640, kernel_size=1),
            nn.BatchNorm1d(640)
        )
        self.block4_continues = block_end(0.3)


        self.block5 = nn.Sequential(
            conv_block(640, 768, 0.3, kernel_size=25, padding=25 // 2 ),
            conv_block(768, 768, 0.3, kernel_size=25, padding=25 // 2 ),
            conv_block(768, 768, 0.3, kernel_size=25, padding=25 // 2 ),
            conv_block(768, 768, 0.3, kernel_size=25, padding=25 // 2 ),
            conv_block(768, 768, 0.3, kernel_size=25, padding=25 // 2 )
        )
        self.one_to_one_conv_5 = nn.Sequential(
            nn.Conv1d(640, 768, kernel_size=1),
            nn.BatchNorm1d(768)
        )
        self.block5_continues = block_end(0.3)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=896,  kernel_size=29, dilation=2, padding=29 // 2 ),
            nn.BatchNorm1d(896),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=896, out_channels=1024,  kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=vocab,  kernel_size=1),
            nn.BatchNorm1d(vocab), #
            nn.ReLU(),
        )

    def forward(self, x):
        # encoder
        e0 = self.conv1(x)

        block_1_left_way = self.block1(e0)
        block_1_alter_right_way = self.one_to_one_conv_1(e0)
        block_1 = self.block1_continues(block_1_left_way + block_1_alter_right_way)
        residual_block_1 = block_1


        block_2_left_way = self.block2(residual_block_1)
        block_2_alter_right_way = self.one_to_one_conv_2(residual_block_1)
        block_2 = self.block2_continues(block_2_left_way + block_2_alter_right_way)
        residual_block_2 = block_2


        block_3_left_way = self.block3(residual_block_2)
        block_3_alter_right_way = self.one_to_one_conv_3(residual_block_2)
        block_3 = self.block3_continues(block_3_left_way + block_3_alter_right_way)
        residual_block_3 = block_3


        block_4_left_way = self.block4(residual_block_3)
        block_4_alter_right_way = self.one_to_one_conv_4(residual_block_3)
        block_4 = self.block4_continues(block_4_left_way + block_4_alter_right_way)
        residual_block_4 = block_4

        block_5_left_way = self.block5(residual_block_4)
        block_5_alter_right_way = self.one_to_one_conv_5(residual_block_4)
        block_5 = self.block5_continues(block_5_left_way + block_5_alter_right_way)
        residual_block_5 = block_5

        e1 = self.conv2(residual_block_5)
        e2 = self.conv3(e1)
        e3 = self.conv4(e2)


        return e3.log_softmax(dim=1)





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
