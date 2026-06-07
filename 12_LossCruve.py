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
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flat(x)
        x = self.fc(x)

        return x
    
def Train_model(model,train_loader,epoch=3,lr=0.01,is_scheduler=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    if is_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.2)

    epoch_losses = []
    batch_losses = []
    print(f'lr:{lr}')   
    for i in range (epoch):
        correct = 0
        total = 0
        accuracy = 0
        
        run_losses = 0
        avr_loss = 0

        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred,y)

            batch_losses.append(loss.item())

            pred_class = pred.argmax(dim=1)
            correct += (pred_class==y).sum().item()
            run_losses += loss.item()
            total += y.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avr_loss = run_losses/len(train_loader)
        accuracy = correct/total
        print(f'In the {i+1} training, the average loss is {avr_loss:.4f}, and the accuracy is {accuracy:.4f}.')
        epoch_losses.append(avr_loss)
        if is_scheduler:
            scheduler.step()

    return epoch_losses,batch_losses

torch.manual_seed(21)
model1 = Simple_CNN().to(device)
torch.manual_seed(21)
model2 = Simple_CNN().to(device)
model3 = Simple_CNN().to(device)

eloss1,bloss1 = Train_model(model1,train_loader,epoch=5,lr=0.01,is_scheduler=True)
eloss2,bloss2 = Train_model(model2,train_loader,epoch=5,lr=0.01,is_scheduler=False)
# eloss3,bloss3 = Train_model(model3,train_loader,3,0.001)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(bloss1, label='schedule')
plt.plot(bloss2, label='normal')
plt.title('Batch Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(eloss1, 'o-', label='schedule')
plt.plot(eloss2, 'o-', label='normal')
plt.title('Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# plt.plot(eloss1, label="lr=0.1")
# plt.plot(eloss2, label="lr=0.01")
# plt.plot(eloss3, label="lr=0.001")
# ex2 = [i*len(train_loader) for i in range (len(eloss2))]
# plt.plot(ex2,eloss2,label='epoch losses')
# plt.plot(bloss2,label='batch losses')

# plt.legend()
# plt.title("Loss vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()