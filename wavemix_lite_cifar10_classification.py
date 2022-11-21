from wavemix_lite import WaveMixLiteImageClassification

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
# Model parameters
num_class=10
num_block=6
dim_channel=64
mul_factor=2
dropout=0.5
# Training parameters
epochs = 100
lr=1e-3
betas=(0.9, 0.999)
eps=1e-8
weight_decay=1e-2
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
train_data = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the test data
test_data = datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=download, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# Set the model
model = WaveMixLiteImageClassification(num_class=num_class,
                                       num_block=num_block,
                                       dim_channel=dim_channel,
                                       mul_factor=mul_factor,
                                       dropout=dropout,
                                       device=device).to(device)
print(model, device)

# Set the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),
                        lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)

# Training
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # Transfer data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Model inference
        outputs = model(inputs)
        
        # Training
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_period == 0 and i != 0:    # print every 2000 mini-batches
            print(f'[{epoch}, {i:5d}] loss: {running_loss / log_period:.3f}')
            running_loss = 0.0

print('Finished Training')