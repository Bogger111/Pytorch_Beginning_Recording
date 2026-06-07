import torch

x = torch.randn(100, 3)   

true_w = torch.tensor([[2.0], [-1.0], [3.0]])
true_b = torch.tensor([0.5])

y = x @ true_w + true_b   

w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.01

for i in range(500):

    y_pred = x @ w + b

    loss = ((y_pred - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if i % 50 == 0:
        print(f"step {i}, loss = {loss.item()}")

print("真实 w:\n", true_w)
print("学到的 w:\n", w)

print("真实 b:", true_b.item())
print("学到的 b:", b.item())