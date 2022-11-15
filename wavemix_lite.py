import torch.nn.functional as F
import torch.nn as nn

class WaveMixLite(nn.Module):
    def __init__(self, block_num=7, channel_dim=16, dropout=0.5):
        super(WaveMixLite, self).__init__()
        # Set the number of blocks
        self.block_num = block_num

        # Set the wave mix lite block
        self.wavemixlite = nn.ModuleList([WaveMixLiteBlock(channel_dim=channel_dim, dropout=dropout) for _ in range(self.block_num)])

    def forward(self, x):
        # Wave mix lite blocks
        for i in range(self.block_num):
            x = self.wavemixlite[i](x)
        
        return x


class WaveMixLiteBlock(nn.Module):
    def __init__(self, channel_dim=16, dropout=0.5):
        super(WaveMixLiteBlock, self).__init__()
        # Set the channel number
        self.channel_dim = channel_dim

        # Set the layers
        self.conv_input = nn.Conv2d(self.channel_dim, self.channel_dim / 4, 1, padding='same')
        self.twod_dwt = twoDDWT()
        self.fc = nn.Conv2d(self.channel_dim, self.channel_dim, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.conv_trans = nn.ConvTranspose2d(self.channel_dim, self.channel_dim, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.batch_norm = nn.BatchNorm2d(self.channel_dim)

    def forward(self, x):
        # Set the residual
        residual = x

        # Network
        x = self.conv_input(x)
        x = self.twod_dwt(x)
        x = self.fc(x)
        x = self.conv_trans(x)
        x = self.batch_norm(x)

        # Residual connection
        x = x + residual

        return x


class twoDDWT(nn.Module):
    def __init__(self):
        super(twoDDWT, self).__init__()
        
    def forward(self, x):
        return x