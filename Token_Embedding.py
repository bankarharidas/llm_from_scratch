import torch
import torch.nn as nn

# Token embedding
embedding = nn.Embedding(num_embeddings=100, embedding_dim=16)

tokens = torch.tensor([5, 12, 27])

x = embedding(tokens)

print("Token embedding shape:", x.shape)


# Position embedding
position_embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)

positions = torch.tensor([0, 1, 2])

pos = position_embedding(positions)

print("Position embedding shape:", pos.shape)


# Combine
x = x + pos

print("Combined shape:", x.shape)
