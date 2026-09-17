import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_head):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_head = num_head
        self.num_dim = embedding_size // num_head

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        Q = Q.view(x.size(0), self.num_head, self.num_dim)
