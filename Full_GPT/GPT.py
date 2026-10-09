import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# RMSNorm
# =========================


class RMSNorm(nn.Module):

    def __init__(self, embedding_size, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(embedding_size))

        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.weight * (x / rms)


# ============================
# Rotary Positional Embedding
# ============================


def apply_rope(x, start_position=0):

    # x shape:
    # [B, num_heads, T, head_dim]

    B, H, T, D = x.shape

    assert D % 2 == 0

    half_dim = D // 2

    # Frequencies
    theta = 10000.0 ** (
        -torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim
    )

    # Position indices
    positions = torch.arange(
        start_position, start_position + T, device=x.device, dtype=torch.float32
    )

    # [T, half_dim]
    angles = positions[:, None] * theta[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # [1, 1, T, half_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotated


# =========================
# Multi-Head Attention
# =========================


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

    def forward(self, x, past_k=None, past_v=None, start_position=0):

        # x: [B, T, C]
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

        # RoPE
        Q = apply_rope(Q, start_position=start_position)

        K = apply_rope(K, start_position=start_position)

        # =========================
        # KV Cache
        # =========================

        if past_k is not None:
            K = torch.cat([past_k, K], dim=2)

        if past_v is not None:
            V = torch.cat([past_v, V], dim=2)

        new_k = K
        new_v = V

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        scores = scores.masked_fill(~mask, float("-inf"))

        # Attention probabilities
        weights = torch.softmax(scores, dim=-1)

        # Attention output
        output = weights @ V

        output = output.transpose(1, 2)

        output = output.contiguous().view(B, T, C)

        output = self.W_O(output)

        return output, new_k, new_v


# =========================
# SwiGLU
# =========================


class SwiGLU(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.gate = nn.Linear(embedding_size, hidden_size)

        self.up = nn.Linear(embedding_size, hidden_size)

        self.down = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        gate = F.silu(self.gate(x))

        up = self.up(x)

        x = gate * up

        x = self.down(x)

        return x


# =========================
# Transformer Block
# =========================


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, num_heads, ffn_hidden_size):
        super().__init__()

        self.norm1 = RMSNorm(embedding_size)

        self.attention = MultiHeadAttention(embedding_size, num_heads)

        self.norm2 = RMSNorm(embedding_size)

        self.ffn = SwiGLU(embedding_size, ffn_hidden_size)

    def forward(self, x, past_k=None, past_v=None, start_position=0):

        # Attention + residual
        attention_output, new_k, new_v = self.attention(
            self.norm1(x), past_k=past_k, past_v=past_v, start_position=start_position
        )

        x = x + attention_output

        # FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x, new_k, new_v


# =========================
# MiniGPT
# =========================


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

    def forward(self, tokens, past_kv=None, start_position=0):

        # tokens: [B, T]
        B, T = tokens.shape

        assert T <= self.sequence_length

        # [B, T] -> [B, T, C]
        x = self.token_embedding(tokens)

        # for block in self.blocks:
        #     x = block(x)

        # Transformer blocks
        new_kv = []

        for layer_index, block in enumerate(self.blocks):

            if past_kv is not None:
                past_k, past_v = past_kv[layer_index]
            else:
                past_k = None
                past_v = None

            x, new_k, new_v = block(
                x, past_k=past_k, past_v=past_v, start_position=start_position
            )

            new_kv.append((new_k, new_v))

        # Final normalization
        x = self.final_norm(x)

        # [B, T, C] -> [B, T, vocab_size]
        logits = self.output_layer(x)

        return logits, new_kv
