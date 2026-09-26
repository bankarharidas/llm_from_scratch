import torch
import torch.nn as nn

batch_size = 4
sequence_length = 8
embedding_size = 128

tokens = torch.randint(0, 1000, (batch_size, sequence_length))

print("Token IDs:", tokens.shape)

token_embedding = nn.Embedding(1000, embedding_size)

print(token_embedding)

position_embedding = nn.Embedding(sequence_length, embedding_size)


x = token_embedding(tokens)

positions = torch.arange(sequence_length)
print("position of embedding", positions)

pos = position_embedding(positions)

x = x + pos

print("Token embedding:", x.shape)
