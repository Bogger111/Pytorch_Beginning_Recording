import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


class CAM_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        feature_map = x

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_map


def get_cam(feature_map, fc_weight, class_idx):
    weight = fc_weight[class_idx]
    cam = torch.zeros(feature_map.shape[2:], dtype=torch.float32, device=feature_map.device)

    for i in range(len(weight)):
        cam += weight[i] * feature_map[0, i, :, :]

    return cam.detach().cpu().numpy()


model = CAM_CNN().to(device)
model.eval()

images, labels = next(iter(test_loader))
image = images[0:1].to(device)
label = labels[0].item()

with torch.no_grad():
    output, feature_map = model(image)

pred = output.argmax(dim=1).item()
fc_weight = model.fc.weight.data
cam = get_cam(feature_map, fc_weight, pred)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[0].cpu().squeeze(), cmap='gray')
plt.title(f'Label: {label}, Pred: {pred}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cam, cmap='jet')
plt.title('CAM')
plt.colorbar()

plt.show()