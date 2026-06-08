import torch
import torch.nn as nn
import math


class MySelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d = d_model

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

    def forward(self, x):

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)

        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)

        return out, weights

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = self.W_o(out)

        return out, weights 


'''
test
'''
x = torch.randn(2, 4, 8)
attn = MyMultiHeadAttention(8, 2)

out, weights = attn(x)

print(out.shape)
print(weights.shape)

print(weights[0])