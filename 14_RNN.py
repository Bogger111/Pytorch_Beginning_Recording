import torch
import torch.nn as nn
import torch.nn.functional as F

text = """
hello world
hello python
hello RNN
deep learning is good
rnn can be run
"""

text = text.strip()

chars = sorted(list(set(text)))

char_to_idx = {ch:idx for idx,ch in enumerate(chars)}

idx_to_char = {idx:ch for ch,idx in char_to_idx.items()}

vocab_size = len(chars)

