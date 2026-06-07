import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-1, 1, 500).reshape(-1, 1)
y = torch.sin(x)

model = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

for i in range(1000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % 100 == 0:
        print(f"step {i}, loss = {loss.item():.6f}")

with torch.no_grad():
    pred = model(x)

plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), y.numpy(), label="true")
plt.plot(x.numpy(), pred.numpy(), label="pred")
plt.legend()
plt.title("Sin(x) Fitting")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()