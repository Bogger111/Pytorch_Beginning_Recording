import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',device)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)

d = iter(train_data)
images,labels = next(d)
img = images.unsqueeze(0)
img = img.to(device)

class Simple_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*7*7,10)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten(start_dim=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x1 = self.relu(x)
        x = self.pool(x1)

        x = self.conv2(x)
        x2 = self.relu(x)
        x = self.pool(x2)

        x = self.flat(x)
        x = self.fc(x)

        return x,x1,x2

gradients = None
def my_save_grad(grad):
    global gradients
    gradients = grad

def cam_generate(model,img):
    model.eval()

    img = img.clone().detach().to(device)
    img.require_grad = True

    output,_,featuremap = model(img)

    featuremap.register_hook(my_save_grad)

    pred_class = output.argmax(dim=1)

    loss = output[0, pred_class.item()]

    model.zero_grad()
    loss.backward()

    weights = gradients.mean(dim=[2, 3], keepdim=True)

    cam = (weights * featuremap).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu()

    cam = cam / cam.max()

    return cam

def show_cam_on_image(img, cam):
    img = img.squeeze().cpu()
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('image')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cam, cmap='jet')
    plt.title("cam")
    plt.axis('off')
    plt.show()

modle1 = Simple_CNN().to(device)
cam = cam_generate(modle1,img)
show_cam_on_image(img,cam)