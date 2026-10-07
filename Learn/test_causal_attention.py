import torch
import torch.nn as nn


class CasualSelfAttention(nn.Module):

    def __init__(self, embedding_size) -> None:
        super().__init__()

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.T

        mask = torch.tril(torch.ones(x.size(0), x.size(0)))

        print("Scores BEFORE mask:")
        print(scores)

        scores = scores.masked_fill(mask == 0, float("-inf"))

        print("\nScore AFTER mask:")
        print(scores)

        weights = torch.softmax(scores, dim=-1)

        print("\nAttention Score:")
        print(weights)

        output = weights @ V

        return output


model = CasualSelfAttention(2)

x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

output = model(x)

print("\nOutput:")
print(output)
