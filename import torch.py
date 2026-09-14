import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_head):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_head = num_head
        self.num_dim = embedding_size // num_head

        W_Q = nn.Linear(embedding_size, embedding_size)
        W_K = nn.Linear(embedding_size, embedding_size)
        W_V = nn.Linear(embedding_size, embedding_size)

        W_O = nn.Linear(embedding_size, embedding_size)

        def forward(self, x):
            Q = self.W_Q(x)
            K = self.W_K(x)
            V = self.W_V(x)

            Q = Q.view(x.size(0), self.num_head, self.num_dim)
            K = K.view(x.size(0), self.num_head, self.num_dim)
            V = V.view(x.size(0), self.num_head, self.num_dim)

            Q = Q.transpose(0, 1)
            K = K.transpose(0, 1)
            V = V.transpose(0, 1)

            score = Q @ K.transpose(-2, -1)

            mask = torch.tril(torch.ones(x.size(0), x.size(0)))

            score = score.masked_fill(mask == 0, float("-inf"))

            weight = torch.softmax(score, dim=-1)

            head_outputs = weight @ V

            head_outputs = head_outputs.transpose(0, 1)

            head_outputs = head_outputs.contiguous().view(
                x.size(0), self.embedding_size
            )

            output = self.W_O(head_outputs)

            return output


class transformer(nn.Module):
    def __init__(self, embedding_size, num_head):
        super().__init__()
