import torch

x = torch.linspace(0,10,50).reshape(-1,1)
y = 3*x + 4 

k = torch.randn(1,requires_grad=True)
b = torch.randn(1,requires_grad=True)

l = 0.01

for i in range(5000):
    y_pred = k*x + b
    loss = ((y_pred - y)**2).mean()

    loss.backward()

    with torch.no_grad():
        k -= l * k.grad
        b -= l * b.grad

        k.grad.zero_()
        b.grad.zero_()

    if i % 500 == 0:
        print(f"step {i},loss = {loss.item()}")
print(k.item())
print(b.item())
