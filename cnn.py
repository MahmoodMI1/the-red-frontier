import torch
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




