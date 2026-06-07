plt.legend()
plt.show()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

x = torch.linspace(-1,1,500).reshape(-1,1)
y = torch.sin(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,20)
        self.fc2 = nn.Linear(20,40)
        self.fc3 = nn.Linear(40,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model1 = Net()
model2 = copy.deepcopy(model1)
optimizer1 = torch.optim.SGD(model1.parameters(),lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(),lr=0.1)
criterion = nn.MSELoss()
losses1 = []
losses2 = []
for i in range (1000):
    pred1 = model1(x)
    pred2 = model2(x)
    l1 = criterion(pred1,y)
    l2 = criterion(pred2,y)

    optimizer1.zero_grad()
    l1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    l2.backward()
    optimizer2.step()
    losses1.append(l1.item())
    losses2.append(l2.item())
    if i % 100 == 0:
        print(f"The loss1 is {l1.item()},in {i} times test")
        print(f"The loss2 is {l2.item()},in {i} times test")

with torch.no_grad():
    pred1 = model1(x)
    pred2 = model2(x)
plt.figure(figsize=(8,6))
# plt.plot(x.numpy(),y.numpy(),label="True")
# plt.plot(x.numpy(),pred.numpy(),label="Pre")
plt.plot(losses1,label="1")
plt.plot(losses2,label="2")
plt.legend()
plt.show()