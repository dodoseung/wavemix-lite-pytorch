from wavemix_lite import WaveMixLiteSemanticSegmentation

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets

from typing import Any, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/cityscapes_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform=A.Compose([A.Resize(1024, 2048),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2()])

class dataset(datasets.Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=transform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']

void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [255, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))

def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = 255
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


# Dataset download
# https://www.cityscapes-dataset.com/downloads/
# gtFine_trainvaltest.zip (241MB)
# leftImg8bit_trainvaltest.zip (11GB)

# Set the training data
train_data = dataset(config['data']['data_path'], split='train', mode='fine', target_type='semantic', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last']
                                           , pin_memory=True)

# Set the test data
val_data = dataset(config['data']['data_path'], split='val', mode='fine', target_type='semantic', transform=transform)
val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=config['data']['batch_size'],
                                        shuffle=config['data']['shuffle'],
                                        num_workers=config['data']['num_workers'],
                                        drop_last=config['data']['drop_last']
                                        , pin_memory=True)

# Check the categories
print(len(train_data.classes))

# Set the model
model = WaveMixLiteSemanticSegmentation(num_class=config['model']['num_class'],
                                       num_block=config['model']['num_block'],
                                       dim_channel=config['model']['dim_channel'],
                                       mul_factor=config['model']['mul_factor'],
                                       dropout=config['model']['dropout'],
                                       device=device).to(device)
# model = nn.DataParallel(model)
print(model, device)

# Set the loss function
class FocalLoss2d(nn.Module):
  def __init__(self, gamma=2, alpha=None):
    super(FocalLoss2d, self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    
  def forward(self, input, target):
    # Input reshape
    input = input.contiguous().view(input.size(0), input.size(1), -1)
    input = input.transpose(1,2)
    input = input.contiguous().view(-1, input.size(2)).squeeze()

    # Target reshape
    target = target.view(-1)
    
    # Calculate the focal loss
    logpt = - F.cross_entropy(input, target)
    pt = torch.exp(logpt)
    focal_loss = - ((1 - pt) ** self.gamma) * logpt
    
    return focal_loss.mean()
    
# Set the criterion and optimizer
criterion = FocalLoss2d()
optimizer = optim.AdamW(model.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'],
                        eps=config['train']['eps'],
                        weight_decay=config['train']['weight_decay'])

# Training
def train(epoch, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0.0
  train_num = 0
  
  with tqdm(train_loader, unit="batch") as tepoch:
    i = 0
    for data in tepoch:
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
        
      # Transfer data to device
      inputs = inputs.to(device)
      labels = labels.to(device)
      segment = encode_segmap(labels)
      
      # Model inference
      outputs = model(inputs)
      
      # Training
      optimizer.zero_grad()
      loss = criterion(outputs, segment.long())
      loss.backward()
      optimizer.step()

      # loss
      train_loss += loss.item()
      train_num += labels.size(0)
      
      if i % config['others']['log_period'] == 0 and i != 0:
        print(f'[{epoch}, {i}]\t Train loss: {train_loss / train_num:.3f}')
      i = i + 1
  
  # Average loss
  train_loss /= train_num
  
  return train_loss

# Validation
def valid(val_loader):
  model.eval()
  corrects = 0
  test_num = 0

  with tqdm(val_loader, unit="batch") as tepoch:
    for data in tepoch:
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
  for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
    # Training
    train_loss = train(epoch, train_loader, optimizer, criterion)
    
    # Validation
    test_accuracy = valid(val_loader)
    
    # Print the log
    print(f'Epoch: {epoch}\t Train loss: {train_loss:.3f}\t Valid accuracy: {test_accuracy:.3f}')
    
    # Save the model
    save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=optimizer, loss=train_loss, config=config)