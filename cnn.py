import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from Hirise import HiRise
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Tweakers
batch_size = 4
num_workers = 4

# root of the data
root = "data/hirise-map-proj-v3_2"

# transformers for both train, and valuation
train_transformer = transforms.Compose([
    transforms.ToTensor()
])
eval_transformer = transforms.Compose([
    transforms.ToTensor()
])



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1) # padding included
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.fc1 = nn.Linear(32*56*56,512)
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


def main(): # main func
    # 80 10 10 split of data sets
    train_dataset = HiRise(root, split="train", transform=train_transformer)
    val_dataset = HiRise(root,split="val", transform=eval_transformer)
    test_dataset = HiRise(root, split="test")

    # data loaders/ shuffle / workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=False)

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()

    lr = 0.01
    num_epochs = 4
    total_steps = len(train_loader)

    #loss tracker
    lossi = []

    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            x = model(images)
            loss = criterion(x,labels)  

            #backward pass
            for p in model.parameters():# parameters()
                p.grad = None
            loss.backward()
            
            for p in model.parameters():
                p.data += -lr * p.grad

            if (i+1) % 2000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

            lossi.append(loss.log10().item())
  



    print('Finished Training')
    #torch.save(model.state_dict(), f'./hirise_cnn_{datetime.now():%Y-%m-%d_%H-%Mwwwwwwwwww}.pth')
    torch.save(model.state_dict(), './hirise_cnn')

    n = len(lossi) // 1000 * 1000 # drop leftover steps so it reshapes evenly
    plt.plot(torch.tensor(lossi[:n]).view(-1,1000).mean(1))
    plt.show()

    
    @torch.no_grad()
    def split_loss(split):
        model.eval()
        x = {
            'train': train_loader,
            'val': val_loader,
            'test':test_loader
        } [split]

        for images, labels in x:
            images = images.to(device)
            labels = labels.to(device)

            x = model(images)
            loss = criterion(x,labels)
        print(split, loss.item())
        #return loss.item()
    
    split_loss('train')
    split_loss('val')   



if __name__ == '__main__':
    main()