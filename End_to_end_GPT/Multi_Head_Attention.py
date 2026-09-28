import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

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

        # x: [B, T, C]
        B, T, C = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # [B, T, C]
        #       ↓
        # [B, T, heads, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        # [B, heads, T, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale
        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        scores = scores.masked_fill(~mask, float("-inf"))

        # Attention probabilities
        weights = torch.softmax(scores, dim=-1)

        # Weighted values
        output = weights @ V

        # [B, heads, T, head_dim]
        #       ↓
        # [B, T, heads, head_dim]
        output = output.transpose(1, 2)

        # Merge heads
        output = output.contiguous().view(B, T, C)

        # Output projection
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

        self.attention = MultiHeadAttention(embedding_size, num_heads)

        self.norm2 = RMSNorm(embedding_size)

        self.ffn = SwiGLU(embedding_size, ffn_hidden_size)

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(self.norm1(x))

        # FFN + residual
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

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        # Position embedding
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, ffn_hidden_size)
                for _ in range(num_blocks)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(embedding_size)

        # LM Head
        self.output_layer = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, tokens):

        # tokens: [B, T]

        B, T = tokens.shape

        assert T <= self.sequence_length

        # Token embeddings
        x = self.token_embedding(tokens)

        # Positions
        positions = torch.arange(T, device=tokens.device)

        pos = self.position_embedding(positions)

        # Add token + position embeddings
        x = x + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Convert to logits
        logits = self.output_layer(x)

        return logits


# -------------------------
# Model configuration
# -------------------------

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
