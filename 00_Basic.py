import torch
print(torch.__version__)
print(torch.rand(2,2))

x = torch.tensor([1,4,6])
print(x)
print(x.shape)

a = torch.tensor(1.0,requires_grad=True)
y = a**2 + a*2
z = torch.sin(y) + torch.log10(a)
z.backward()
print(a.grad)

