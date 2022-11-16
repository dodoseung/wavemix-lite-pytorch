import torch
import torch.nn as nn


class twoDDWT(nn.Module):
    def __init__(self, device):
        super(twoDDWT, self).__init__()
        self.n = torch.tensor(2, dtype=torch.float32, device=device)

    def forward(self, x):
        low, high = self.haar_transform_vertical(x)
        ll, lh = self.haar_transform_horizontal(low)
        hl, hh = self.haar_transform_horizontal(high)

        return ll, lh, hl, hh
    
    def haar_transform_vertical(self, x):
        low = (x[:, :, ::2, :] + x[:, :, 1::2, :]) / torch.sqrt(self.n)
        high = (x[:, :, ::2, :] - x[:, :, 1::2, :]) / torch.sqrt(self.n)

        return low, high

    def haar_transform_horizontal(self, x):
        low = (x[:, :, :, ::2] + x[:, :, :, 1::2]) / torch.sqrt(self.n)
        high = (x[:, :, :, ::2] - x[:, :, :, 1::2]) / torch.sqrt(self.n)
        
        return low, high

class WaveMixLiteBlock(nn.Module):
    def __init__(self, dim_channel=128, mul_factor=2, dropout=0.5, device='cpu'):
        super(WaveMixLiteBlock, self).__init__()
        # Set the channel number
        self.dim_channel = dim_channel

        # Set the layers
        # Work for reduction the parameters and compuations
        self.conv_input = nn.Conv2d(self.dim_channel, int(self.dim_channel/4), kernel_size=(1, 1), stride=(1, 1))
        self.twod_dwt = twoDDWT(device=device)

        # MLP layer (two 1 × 1 convolutional layers separated by a GELU non-linearity)
        self.fc = nn.Sequential(
            nn.Conv2d(self.dim_channel, self.dim_channel * mul_factor, kernel_size=(1, 1), stride=(1, 1)),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.dim_channel * mul_factor, self.dim_channel, kernel_size=(1, 1), stride=(1, 1))
            )

        # Transposed convolution to reconciliate the image size
        self.conv_trans = nn.ConvTranspose2d(self.dim_channel, self.dim_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(self.dim_channel)

    def forward(self, x):
        # Set the residual
        residual = x

        # Input conv layer
        x = self.conv_input(x)

        # 2D-DWT
        ll, lh, hl, hh = self.twod_dwt(x)

        # Concatenate the output images of 2D-DWT
        x = torch.cat((ll, lh, hl, hh), 1)
        
        # MLP
        x = self.fc(x)

        # Transposed conv layer
        x = self.conv_trans(x)

        # Batch normalization
        x = self.batch_norm(x)

        # Residual connection
        x = x + residual

        return x


class WaveMixLite(nn.Module):
    def __init__(self, num_block=8, dim_channel=128, mul_factor=2, dropout=0.5, device='cpu'):
        super(WaveMixLite, self).__init__()
        # Set the number of blocks
        self.num_block = num_block

        # Set the device
        self.device = device

        # Set the wave mix lite block
        self.wavemixlite = nn.ModuleList([WaveMixLiteBlock(dim_channel=dim_channel, mul_factor=mul_factor, dropout=dropout, device=device) for _ in range(self.num_block)])

    def forward(self, x):
        # Wave mix lite blocks
        for i in range(self.num_block):
            x = self.wavemixlite[i](x)
        
        return x


class WaveMixLiteImageClassification(nn.Module):
    def __init__(self, num_class=1000, num_block=8, dim_channel=128, mul_factor=2, dropout=0.5, device='cpu'):
        super(WaveMixLite, self).__init__()
        # Set the wave mix lite network
        self.wavemixlite = WaveMixLite(num_block=num_block, dim_channel=dim_channel, mul_factor=mul_factor, dropout=dropout, device=device)

        # Set the number of classes
        self.num_class = num_class

        # Set the initial conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, int(dim_channel / 2), kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.Conv2d(int(dim_channel / 2), dim_channel, kernel_size=(3, 3), stride=(1, 1), padding='same')
        )

        # Set the MLP head layer
        self.fc = nn.Linear()

        # Set the global average pooling
        self.pool = nn.AvgPool1d(1)

        # Set the softmax function
        self.softmax = nn.Softmax()

    def forward(self, x):

        return x