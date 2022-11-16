from wavemix_lite import WaveMixLite

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Data parameters
batch_size = 256
shuffle = True
drop_last = True
download = True
# Training parameters
epochs = 10
learning_rate = 0.0002
# Network parameters
num_layers = 10
# Other parameters
log_period = 50

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# Set the transform
transform = transforms.Compose([transforms.ToTensor()])

# Set the training data
data_train = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=True, transform=transform)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the test data
data_test = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=False, transform=transform)
loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the model
model = WaveMixLite(block_num=7, channel_dim=16, mul_factor=2, dropout=0.5, device=device).to(device)
print(model, device)