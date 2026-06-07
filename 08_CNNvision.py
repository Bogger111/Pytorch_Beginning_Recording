import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',device)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
d = iter(train_data)
images,labels = next(d)
img = images.unsqueeze(0)
img = img.to(device)

class Simple_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*7*7,10)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten(start_dim=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x1 = self.relu(x)
        x = self.pool(x1)

        x = self.conv2(x)
        x2 = self.relu(x)
        x = self.pool(x2)

        x = self.flat(x)
        x = self.fc(x)

        return x,x1,x2

def Train_model(model,train_datas,criterion,optimizer,epoch=3,lr=0.001):
    model.train()

    run_losses = 0
    avr_loss = 0

    correct = 0
    total = 0
    accuracy = 0

    for i in range (epoch):
        for x,y in train_datas:
            x = x.to(device)
            y = y.to(device)

            pred,_,_ = model(x)
            loss = criterion(pred,y)

            pred_class = pred.argmax(dim=1)
            correct += (pred_class==y).sum().item()
            run_losses += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total += y.size[0]
        avr_loss = run_losses/total
        accuracy = correct/total
        print(f'In the {i+1} training, the average loss is {avr_loss}, and the accuracy is {accuracy}.')


def show_image(featrue,num=8):
    featrue = featrue.detach()
    fig,axes = plt.subplots(1,num,figsize=(15,3))
    
    for i in range (num):
        axes[i].imshow(featrue[0,i].cpu(),cmap='gray')
        axes[i].axis('off')
    
    plt.show()


model = Simple_CNN().to(device)

model.eval()
output,filiter1,filiter2 = model(img)

show_image(filiter1)
show_image(filiter2)