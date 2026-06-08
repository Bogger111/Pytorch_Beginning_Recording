import torch
import torch.nn as nn
import math

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

class MyTransformBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MyMultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)

        x = self.norm2(x + ffn_out)

        return x, attn_weights
    
class MyTokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return token_emb + pos_emb

vocab_size = 20
max_len = 10
d_model = 8

embed = MyTokenPositionEmbedding(vocab_size, max_len, d_model)

x = torch.tensor([
    [1, 5, 9, 2],
    [3, 4, 7, 0]
])

out = embed(x)

print("x shape:", x.shape)
print("out shape:", out.shape)
print(out)

'''
test
'''

# x = torch.randn(2, 4, 8)
# block = MyTransformBlock(8, 2, 32)

# out, weights = block(x)

# print(out.shape)
# print(weights.shape)