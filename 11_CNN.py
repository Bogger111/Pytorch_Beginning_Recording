import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_data = datasets.MNIST(root='./data',train=False,transform=transform,download=True)

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1,16,3,padding = 1)
        self.conv2 = nn.Conv2d(16,32,3,padding = 1)
        self.fc = nn.Linear(32*7*7,10)
        self.flt = nn.Flatten(start_dim=1)
        self.pl = nn.MaxPool2d(2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pl(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pl(x)

        x = self.flt(x)
        x = self.fc(x)
        return x

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range (3):
    model.train()
    losses = 0

    for x , y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        avgloss = losses/len(train_loader)
        
    print(f"{epoch + 1} has done!")
    print(f'The loss is {avgloss:.4f}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x , y in test_loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        pred_class = pred.argmax(dim = 1)
        correct += (pred_class == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy:{correct/total:.2f}")