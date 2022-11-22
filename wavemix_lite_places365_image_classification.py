from wavemix_lite import WaveMixLiteImageClassification

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets, transforms

# Data parameters
batch_size = 256
shuffle = True
drop_last = True
download = True
num_workers = 4
# Model parameters
num_class = 365
num_block = 7
dim_channel = 256
mul_factor = 2
dropout = 0.5
# Training parameters
epochs = 100
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-2
# Other parameters
log_period = 50

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256))])

# Data download
# kaggle datasets download -d benjaminkz/places365

# Set the training data
train_data = datasets.ImageFolder(root='~/.pytorch/PLACES365_data/train',
                                  transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           drop_last=drop_last)

# Set the test data
test_data = datasets.ImageFolder(root='~/.pytorch/PLACES365_data/val',
                                  transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          drop_last=drop_last)

# Check the categories
print(len(train_data.classes))

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
def train(epoch, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0.0
  train_num = 0
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

    # loss
    train_loss += loss.item()
    train_num += 1
    
    if i % log_period == 0 and i != 0:
      print(f'[{epoch}, {i}]\t Train loss: {train_loss / train_num:.3f}')
  
  # Average loss
  train_loss /= train_num
  
  return train_loss

# Validation
def valid(test_loader):
  model.eval()
  corrects = 0
  test_num = 0

  for _, data in enumerate(test_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
      
    # Transfer data to device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Model inference
    outputs = model(inputs)
    
    # Calculate the accuracy
    preds = torch.argmax(outputs.data, 1)
    corrects += torch.sum(preds == labels.data).item()

    # Number of the data
    test_num += labels.size(0)
  
  # Test accuracy
  test_accuracy = 100 * corrects / test_num
  
  return test_accuracy

# Main
if __name__ == '__main__':
  for epoch in range(epochs):  # loop over the dataset multiple times
    # Training
    train_loss = train(epoch, train_loader, optimizer, criterion)
    
    # Validation
    test_accuracy = valid(test_loader)
    
    # Print the log
    print(f'Epoch: {epoch}\t Train loss: {train_loss:.3f}\t Valid accuracy: {test_accuracy:.3f}')