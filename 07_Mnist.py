import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data",train=True,transform=transform,download=True)
tests_data = datasets.MNIST(root="./data",train=False,transform=transform,download=True)

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
tests_loader = DataLoader(tests_data,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,100)
        self.fc2 = nn.Linear(100,200)
        self.fc3 = nn.Linear(200,10)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = x.reshape(-1,28*28)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()    
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
criterion = nn.CrossEntropyLoss()


for epoch in range (3):
    for x,y in train_loader:
        pred = model(x)
        loss = criterion(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{epoch + 1} has done")

correct = 0
total = 0

plt.figure(figsize=(10,6))

count = 0

with torch.no_grad():
    for x,y in tests_loader:
        pred = model(x)
        pred_class = pred.argmax(dim=1)
        correct += (pred_class == y).sum().item()
        total += y.size(0)

    # for i in range(len(x)):
    #     if y[i] != pred_class[i]:
    #         plt.subplot(2,5,count+1)
    #         plt.imshow(x[i].squeeze(), cmap='gray')
    #         plt.title(f"T:{y[i]} P:{pred_class[i]}", color="red")
    #         plt.axis('off')
    #         count += 1
    #         if count == 10:
    #             break

print(f"Test accuracy is {correct/total:.4f}")



for i in range (10):
    plt.subplot(2,5,i+1)
    plt.imshow(x[i].squeeze(),cmap="gray")
    color = "green" if y[i]==pred_class[i] else "red"
    plt.title(f"T:{y[i]} P:{pred_class[i]}", color=color)
    plt.axis('off')

plt.show()