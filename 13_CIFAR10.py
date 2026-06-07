import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

transform =transforms.ToTensor()
train_datas = datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_datas = datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)
train_loaders = DataLoader(train_datas,batch_size=256,shuffle=True)
test_loaders = DataLoader(test_datas,batch_size=256,shuffle=False)

configs = {
    # "nodrop,nosche": {"dropout": False, "scheduler": False},
    # "nodrop,sche":   {"dropout": False, "scheduler": True},
    # "drop,nosche":   {"dropout": True,  "scheduler": False},
    "drop,sche":     {"dropout": True,  "scheduler": True}
}
models = {}
class SimpleNet(nn.Module):
    def __init__(self,use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1)
        )
        self.flt = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(256*4*4,256),
            nn.ReLU()
        )
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        self.lin = nn.Linear(256,10)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.flt(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x
    
def model_trainer(model,train_loader,epoch=3,lr=0.01,use_scheduler=False):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.8)

    elosses = []
    blosses = []
    accuracies = []
    model.train()
    for i in range (epoch):
        correct = 0
        total = 0
        run_losses = 0

        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred,y)
            pred_class = pred.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (pred_class == y).sum().item()
            total += y.size(0)

            run_losses += loss.item()

            blosses.append(loss.item())
            accuracies.append((pred_class == y).sum().item()/y.size(0))
        acc = correct/total
        avr_loss = run_losses/len(train_loader)
        elosses.append(avr_loss)
        if (i+1)%10 == 0 :
            print(f"Times: {i+1}, Loss: {avr_loss}, Accuracy: {acc}.")

        if use_scheduler:
            scheduler.step()

    return elosses,blosses,accuracies

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
    results = {}
    for name, cfg in configs.items():
        torch.manual_seed(24)

        model = SimpleNet(cfg["dropout"]).to(device)
        models[name] = model
        print(name)
        eloss, bloss, acc = model_trainer(
            model,
            train_loaders,
            epoch=20,
            lr=0.001,
            use_scheduler=cfg["scheduler"]
        )

        test_acc = model_test(model,test_loaders)
        print(f"{name} test acc: {test_acc:.4f}")

        results[name] = (eloss, bloss, acc)
    return results

def plot_results(results):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    for label, (eloss, bloss, acc) in results.items():
        plt.plot(eloss, label=label)
    plt.title("Epoch Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    for label, (eloss, bloss, acc) in results.items():
        plt.plot(bloss, label=label)
    plt.title("Batch Loss")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    for label, (eloss, bloss, acc) in results.items():
        plt.plot(acc, label=label)
    plt.title("Accuracy")
    plt.xlabel("batch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

results = model_running(configs)
plot_results(results)

