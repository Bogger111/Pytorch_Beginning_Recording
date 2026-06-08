"""
Six-model RNN family comparison for a tiny sentiment classification task.
Models:
1. MyRNN   - handwritten RNNCell
2. MyLSTM  - handwritten LSTMCell
3. MyGRU   - handwritten GRUCell
4. TorchRNN  - nn.RNN
5. TorchLSTM - nn.LSTM
6. TorchGRU  - nn.GRU

Run:
    python rnn_lstm_gru_compare.py

Outputs:
    results_rnn_compare/loss_curves.png
    results_rnn_compare/acc_curves.png
    results_rnn_compare/final_test_acc.png
    results_rnn_compare/comparison.csv
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from datasets import load_dataset

torch.set_num_threads(1)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_rotten_tomatoes_dataset() -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load HuggingFace rotten_tomatoes dataset and return train/test text-label lists."""
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

    train_sentences = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_sentences = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    return train_sentences, train_labels, test_sentences, test_labels


def build_vocab(sentences: List[str]) -> Dict[str, int]:
    word2idx = {"<pad>": 0, "<unk>": 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def encode(sentence: str, word2idx: Dict[str, int], max_len: int) -> List[int]:
    ids = [word2idx.get(word, word2idx["<unk>"]) for word in sentence.split()]
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]


def make_tensors(sentences: List[str], labels: List[int], word2idx: Dict[str, int], max_len: int):
    x = torch.tensor([encode(s, word2idx, max_len) for s in sentences], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.input_hidden(x_t) + self.hidden_hidden(h_prev))
        return h_t


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
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
        return h_t, c_t


class MyGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x_z = nn.Linear(input_size, hidden_size)
        self.h_z = nn.Linear(hidden_size, hidden_size)
        self.x_r = nn.Linear(input_size, hidden_size)
        self.h_r = nn.Linear(hidden_size, hidden_size)
        self.x_h = nn.Linear(input_size, hidden_size)
        self.h_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev))
        r_t = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))
        h_tilde = torch.tanh(self.x_h(x_t) + self.h_h(r_t * h_prev))
        h_t = (1 - z_t) * h_tilde + z_t * h_prev
        return h_t


class MyRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell = MyRNNCell(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


class MyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell = MyLSTMCell(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
        return self.fc(h)


class MyGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell = MyGRUCell(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


class TorchRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        return self.fc(h_n[-1])


class TorchLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1])


class TorchGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        return self.fc(h_n[-1])


@dataclass
class Config:
    seed: int = 42
    max_len: int = 40
    embed_dim: int = 16
    hidden_size: int = 16
    num_classes: int = 2
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-2
    output_dir: str = "results_rnn_compare"


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_one_model(name, model_class, vocab_size, train_loader, test_loader, cfg, device):
    set_seed(cfg.seed)  
    model = model_class(vocab_size, cfg.embed_dim, cfg.hidden_size, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(cfg.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"{name:10s} | Epoch {epoch + 1:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    return model, history



def plot_curves(histories: Dict[str, Dict[str, List[float]]], metric: str, title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(range(1, len(hist[metric]) + 1), hist[metric], label=name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_final_test_acc(histories: Dict[str, Dict[str, List[float]]], save_path: str):
    names = list(histories.keys())
    accs = [histories[name]["test_acc"][-1] for name in names]
    plt.figure(figsize=(10, 6))
    plt.bar(names, accs)
    plt.title("Final Test Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_csv(histories: Dict[str, Dict[str, List[float]]], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("model,epoch,train_loss,train_acc,test_loss,test_acc\n")
        for name, hist in histories.items():
            for i in range(len(hist["train_loss"])):
                f.write(
                    f"{name},{i + 1},"
                    f"{hist['train_loss'][i]:.6f},"
                    f"{hist['train_acc'][i]:.6f},"
                    f"{hist['test_loss'][i]:.6f},"
                    f"{hist['test_acc'][i]:.6f}\n"
                )



def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_sentences, train_labels, test_sentences, test_labels = build_rotten_tomatoes_dataset()

    word2idx = build_vocab(train_sentences)
    vocab_size = len(word2idx)

    x_train, y_train = make_tensors(train_sentences, train_labels, word2idx, cfg.max_len)
    x_test, y_test = make_tensors(test_sentences, test_labels, word2idx, cfg.max_len)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    print(f"Train size: {len(train_sentences)}, Test size: {len(test_sentences)}")
    print(f"Vocab size: {vocab_size}")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    model_classes = {
        "MyRNN": MyRNN,
        "MyLSTM": MyLSTM,
        "MyGRU": MyGRU,
        "nn.RNN": TorchRNN,
        "nn.LSTM": TorchLSTM,
        "nn.GRU": TorchGRU,
    }

    histories = {}
    trained_models = {}

    for name, model_class in model_classes.items():
        print("\n" + "=" * 70)
        print(f"Training {name}")
        print("=" * 70)
        model, history = train_one_model(name, model_class, vocab_size, train_loader, test_loader, cfg, device)
        histories[name] = history
        trained_models[name] = model

    loss_path = os.path.join(cfg.output_dir, "loss_curves.png")
    acc_path = os.path.join(cfg.output_dir, "acc_curves.png")
    final_acc_path = os.path.join(cfg.output_dir, "final_test_acc.png")
    csv_path = os.path.join(cfg.output_dir, "comparison.csv")

    plot_curves(histories, "test_loss", "Test Loss Curves", "Loss", loss_path)
    plot_curves(histories, "test_acc", "Test Accuracy Curves", "Accuracy", acc_path)
    plot_final_test_acc(histories, final_acc_path)
    save_csv(histories, csv_path)

    print("\n" + "=" * 70)
    print("Final result")
    print("=" * 70)
    for name, hist in histories.items():
        print(
            f"{name:10s} | "
            f"final_train_acc={hist['train_acc'][-1]:.4f} | "
            f"final_test_acc={hist['test_acc'][-1]:.4f} | "
            f"final_test_loss={hist['test_loss'][-1]:.4f}"
        )

    print("\nSaved files:")
    print(f"- {loss_path}")
    print(f"- {acc_path}")
    print(f"- {final_acc_path}")
    print(f"- {csv_path}")

    demo_model = trained_models["nn.GRU"]
    demo_model.eval()
    demo_sentences = [
        "i love this movie",
        "this movie is terrible",
        "this story is wonderful",
        "i dislike this film",
        "this game is amazing",  
    ]
    print("\nPrediction demo using nn.GRU:")
    with torch.no_grad():
        for s in demo_sentences:
            xx = torch.tensor([encode(s, word2idx, cfg.max_len)], dtype=torch.long).to(device)
            logits = demo_model(xx)
            pred = logits.argmax(dim=1).item()
            label = "positive" if pred == 1 else "negative"
            print(f"{s:30s} -> {label}")


if __name__ == "__main__":
    main()
