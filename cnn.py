import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from Hirise import HiRise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Tweakers
batch_size = 4
num_workers = 4

# root of the data
root = "data/hirise-map-proj-v3_2/hirise-map-proj-v3_2"

# transformers for both train, and valuation
train_transformer = transforms.Compose([
    transforms.ToTensor()
])
eval_transformer = transforms.Compose([
    transforms.ToTensor()
])

# 80 10 10 split of data sets
train_dataset = HiRise(root, split="train")
val_dataset = HiRise(root,split="val")
test_dataset = HiRise(root, split="test")

# data loaders/ shuffle / workers
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size,
                                            num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size,
                                            num_workers, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size,
                                            num_workers,shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3,16,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.fc1 = nn.Linear(32*56*56,256)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,8)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,32*56*56)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x