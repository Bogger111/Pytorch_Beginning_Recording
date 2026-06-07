import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt

x = torch.randn(100,2)
r = x[:,0]**2 + x[:,1]**2
theta = torch.atan2(x[:,1], x[:,0])
y = (r < 1 + torch.sin(5*theta)*0.3).long()
noise = torch.rand(len(y)) < 0.1
y[noise] = 1 - y[noise]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,100)
        self.fc2 = nn.Linear(100,400)
        self.fc3 = nn.Linear(400,2)
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x
model = Net()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
criterion = nn.CrossEntropyLoss()

losses = []
for i in range (5000):
    pred = model(x)
    loss = criterion(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 1000 == 0:
        print(f"The loss in {i} test is {loss}")

with torch.no_grad():
    pred = model(x)
    pred_class = pred.argmax(dim=1)

acc = (pred_class == y).float().mean()
print(f"accuracy:{acc.item():.4f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c = y)
plt.title("True Label")

plt.subplot(1,2,2)
plt.scatter(x[:,0],x[:,1],c = pred_class)
plt.title("Model prediction")

plt.show()