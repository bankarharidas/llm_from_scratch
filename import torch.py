import math
import re
from turtle import forward
from cv2 import transpose
from networkx import weakly_connected_components
from numpy import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as F


class MutiHeadAttention(nn.Module):

    def __init__(self, embedding_size, num_heads):
        super().__init__()

        assert embedding_size % num_heads == 0

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)

        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weight = torch.softmax(scores, dim=-1)

        output = weight @ V

        output = output.transpose(1, 2)

        output = output.contiguous().view(B, T, C)

        output = self.W_O(output)

        return output


class RMSNorm(nn.Module):
    def __init__(self, embedding_size, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(embedding_size))

        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.gate = nn.Linear(embedding_size, hidden_size)

        self.up = nn.Linear(embedding_size, hidden_size)

        self.down = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        gate = torch.nn.functional.silu(self.gate(x))

        up = self.up(x)

        x = gate * up

        x = self.down(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, ffn_hidden_size):
        super().__init__()

        self.norm1 = RMSNorm(embedding_size)

        self.attention = MutiHeadAttention(embedding_size, num_heads)

        self.norm2 = RMSNorm(embedding_size)

        self.ffn = SwiGLU(embedding_size, ffn_hidden_size)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        num_heads,
        num_blocks,
        sequence_length,
        ffn_hidden_size,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.block = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, ffn_hidden_size)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = RMSNorm(embedding_size)

        self.output_layer = nn.Linear(embedding_size, vocab_size, bias=True)

    def forward(self, tokens):

        B, T = tokens.shape

        assert T <= self.sequence_length

        x = self.token_embedding(tokens)

        position = torch.arange(T, device=tokens.device)

        pos = self.position_embedding(position)

        x = x + pos

        for block in self.block:

            x = block(x)

        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits


vocab_size = 1000
embedding_size = 128
num_heads = 4
num_blocks = 4
sequence_length = 128
ffn_hidden_size = 512

# -------------------------
# Create model
# -------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    sequence_length=sequence_length,
    ffn_hidden_size=ffn_hidden_size,
)

# -------------------------
# Random training batch
# -------------------------

tokens = torch.randint(0, vocab_size, (8, 32))

inputs = tokens[:, :-1]
targets = tokens[:, 1:]

# -------------------------
# Forward pass
# -------------------------

logits = model(inputs)

# -------------------------
# Loss
# -------------------------

loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

print("Tokens :", tokens.shape)
print("Inputs :", inputs.shape)
print("Targets:", targets.shape)
print("Logits :", logits.shape)
print("Loss   :", loss.item())

# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -------------------------
# Backward + update
# -------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Loss   :", loss.item())
print("Training step completed!")

# -------------------------
# Parameter count
# -------------------------

total_parameters = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_parameters)
