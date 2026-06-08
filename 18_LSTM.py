import torch
import torch.nn as nn
import torch.optim as optim


sentences = [
    "i love this movie",
    "this film is great",
    "i like this story",
    "this movie is wonderful",
    "this film is amazing",
    "i enjoy this movie",
    "this story is good",
    "this movie makes me happy",

    "i hate this movie",
    "this film is bad",
    "i dislike this story",
    "this movie is terrible",
    "this film is awful",
    "i do not like this movie",
    "this story is boring",
    "this movie makes me sad",
]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 
    0, 0, 0, 0, 0, 0, 0, 0,
]

word2idx = {'<pad>':0, '<unk>':1}
for sentence in sentences:
    for word in sentence.split():
        if word not in word2idx:
            word2idx[word] = len(word2idx)

max_len = 5

def encode(sentence):
    ids = []

    for word in sentence.split():
        ids.append(word2idx.get(word, word2idx["<unk>"]))

    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))

    ids = ids[:max_len]

    return ids

x = torch.tensor([encode(s) for s in sentences])
y = torch.tensor(labels)

vocab_size = len(word2idx)
input_size = 16
hidden_size = 10
nums_classes = 2

epoches = 20

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size)

        self.x_i = nn.Linear(input_size, hidden_size)
        self.h_i = nn.Linear(hidden_size, hidden_size)

        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size)

        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        
        f_t = torch.sigmoid(self.x_f(x_t) + self.h_f(h_prev))
        i_t = torch.sigmoid(self.x_i(x_t) + self.h_i(h_prev))
        g_t = torch.tanh(self.x_g(x_t) + self.h_g(h_prev))
        o_t = torch.sigmoid(self.x_o(x_t) + self.h_o(h_prev))
        
        c_t = f_t * c_prev + i_t * g_t

        h_t = o_t * torch.tanh(c_t)
        
        return h_t
    

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nums_classes, vocab_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nums_classes = nums_classes
        

        self.cell = MyLSTMCell(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size,nums_classes)
        self.embedding = nn.Embedding(vocab_size, input_size)

    def forward(self, x):
        
        x = self.embedding(x)
        batch_size, seq_len, input_size = x.shape
        
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.cell(x_t, h, c)

        logits = self.fc(h)

        return logits

def predict(sentence):
    model.eval()

    x_test = torch.tensor([encode(sentence)])

    with torch.no_grad():
        logits = model(x_test)
        pred = logits.argmax(dim=1).item()

        if pred == 1:
            return 'positive'
        else:
            return 'negative'

model = MyLSTM(input_size, hidden_size, nums_classes, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epoches):

    logits = model(x)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean()
    
    if epoch == 0:
        print(f"Epoch{epoch+1}, Loss:{loss.item():.4f}, Acc:{acc.item():.4f}") 
    
    if(epoch + 1) % 10 == 0:
        print(f'Epoch{epoch+1}, Loss:{loss.item():.4f}, Acc:{acc.item():.4f}')

print(predict("i love this story"))
print(predict("this movie is bad"))
print(predict("i hate this film"))
print(predict("this story is damn"))
