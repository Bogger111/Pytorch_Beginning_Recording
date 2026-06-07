import torch
import matplotlib.pyplot as plt

x = torch.linspace(-2,2,200).reshape(-1,1)
y = x ** 2

def train(h):
    w1 = torch.randn(1,h,requires_grad=True)
    b1 = torch.randn(h,requires_grad=True)
    w2 = torch.randn(h,1,requires_grad=True)
    b2 = torch.randn(1,requires_grad=True)

    lr = 0.01

    for i in range(0,10000):
        h = x @ w1 +b1
        h = torch.relu(h)
        y_pre = h @ w2 +b2

        loss = ((y-y_pre)**2).mean()
        loss.backward()

        with torch.no_grad():
            for p in [w1,b1,w2,b2]:
                p -= lr * p.grad
                p.grad.zero_()
    return y_pre.detach()
sizes = [120]

plt.figure(figsize=(8,6))
# plt.plot(x.numpy(),y.numpy(),label='true')

for s in sizes:
    y_pre = train(s)
    plt.plot(x.numpy(),y_pre.numpy(),label=f'{s}')