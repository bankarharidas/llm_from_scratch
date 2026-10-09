import torch
import torch.nn as nn
from LLM.Multi_Head_Attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_size, num_heads)

        self.norm1 = nn.LayerNorm(embedding_size)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
        )

        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):

        # Attention
        attention_output = self.attention(x)

        # Residual connection
        x = x + attention_output

        # LayerNorm
        x = self.norm1(x)

        # Feed Forward Network
        ffn_output = self.ffn(x)

        # Residual connection
        x = x + ffn_output

        # LayerNorm
        x = self.norm2(x)

        return x


# block = TransformerBlock(embedding_size=16, num_heads=4)

# output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# vocab_size = 100
# embedding_size = 16
# num_heads = 4
# sequence_length = 5

# token_embedding = nn.Embedding(vocab_size, embedding_size)

# position_embedding = nn.Embedding(sequence_length, embedding_size)

# block = TransformerBlock(embedding_size, num_heads)

# tokens = torch.tensor([5, 12, 27, 8, 19])

# # Token embeddings
# x = token_embedding(tokens)

# # Position embeddings
# positions = torch.arange(sequence_length)
# pos = position_embedding(positions)

# # Combine token + position information
# x = x + pos

# # Transformer block
# output = block(x)

# # print("Tokens:", tokens.shape)
# # print("Token embeddings:", x.shape)
# # print("Transformer output:", output.shape)
