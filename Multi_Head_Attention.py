import torch
import torch.nn as nn

# -------------------------
# Multi-Head Attention
# -------------------------


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, num_heads):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)

        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(x.size(0), self.num_heads, self.head_dim)

        K = K.view(x.size(0), self.num_heads, self.head_dim)

        V = V.view(x.size(0), self.num_heads, self.head_dim)

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        scores = Q @ K.transpose(-2, -1)

        mask = torch.tril(torch.ones(x.size(0), x.size(0)))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        head_outputs = weights @ V

        head_outputs = head_outputs.transpose(0, 1)

        head_outputs = head_outputs.contiguous().view(x.size(0), self.embedding_size)

        output = self.W_O(head_outputs)

        return output


# attention = MultiHeadAttention(embedding_size=16, num_heads=4)

# layer_norm = nn.LayerNorm(16)

# x = torch.randn(5, 16)

# print("Input:", x.shape)

# attention_output = attention(x)

# print("Attention:", attention_output.shape)

# x = x + attention_output

# print("After residual:", x.shape)

# x = layer_norm(x)

# print("After LayerNorm:", x.shape)

# ffn = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))

# ffn_output = ffn(x)

# print("FFN:", ffn_output.shape)

# x = x + ffn_output

# print("After FFN residual:", x.shape)

# layer_norm2 = nn.LayerNorm(16)

# x = layer_norm2(x)

# print("Final output:", x.shape)
