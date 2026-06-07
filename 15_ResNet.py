import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")

train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))]
)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))]
)

train_data = datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transform)
test_data = datasets.CIFAR10(root='./data',train=False,download=True,transform=test_transform)
train_loader = DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = DataLoader(test_data,batch_size=128,shuffle=False)

configs = {
    'drop,sche':{'dropout':True,'scheduler':True,'lr':0.001,'epoch':100}
    }
models = {}
results = {}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU()
        )

        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        else:
            self.dropout = nn.Identity()

        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)      
        x = self.pool(x)      

        x = self.layer1(x)    
        x = self.layer2(x)    
        x = self.layer3(x)    

        x = self.flat(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

def model_train(model,data_loader,optimizer,criterion):
 
    model.train()

    correct = 0
    total = 0
    run_losses = 0
    
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred,y)
        pred_class = pred.argmax(dim = 1)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        correct += (pred_class == y).sum().item()
        total += y.size(0)

        run_losses += loss.item()
   
    train_acc = correct/total
    avr_loss = run_losses/len(data_loader)
    
    return train_acc,avr_loss
    
def model_test(model,test_loader):

    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            pred_class = pred.argmax(dim=1)

            correct += (pred_class == y).sum().item()
            total += y.size(0)
            
    return correct/total

def model_running(configs):
    for name,cfg in configs.items():
        print(name)

        torch.manual_seed(24)
        model = ResNet(use_dropout=cfg['dropout']).to(device)
        models[name] = model
        
        optimizer = optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)

        losses = []
        train_accs = []
        test_accs = []
        best_acc = 0
        for times in range(cfg['epoch']):
            
            train_acc,avr_loss = model_train(
                model,
                train_loader,
                optimizer,
                criterion,
                )
            
            if cfg['scheduler']:
                scheduler.step()
            test_acc = model_test(model,test_loader)
            
            losses.append(avr_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'model':model.state_dict(),
                    'acc':best_acc,
                },'mini_resnet_cifar10.pth')

            print(f"Epoch:{times+1}, Loss:{avr_loss:.4f}, Train accuracy:{train_acc:.4f}, Test accuracy:{test_acc:.4f}, Best accuracy:{best_acc:.4f}.")
            
            results[name] = (losses,train_accs,test_accs)
            

    return results 

def plot_acc(results,configs):
    for name , cfg in configs.items():
        plt.figure(figsize=(12,6))
        
        epochs = range(1,len(results[name][0])+1)
        
        plt.plot(epochs,results[name][1],label = 'train_acc')
        plt.plot(epochs,results[name][2],label = "test_acc")
        
        plt.title(name)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid()
        plt.show()

def plot_loss(results,configs):
    for name , cfg in configs.items():
        plt.figure(figsize=(8,6))

        epochs = range(1,len(results[name][0])+1)

        plt.plot(epochs,results[name][0],label = 'loss')

        plt.xlabel('Epochs')
        plt.ylabel("Loss")

        plt.grid()
        plt.show()

results = model_running(configs)

plot_acc(results,configs)
plot_loss(results,configs)