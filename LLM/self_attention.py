import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()

        self.W_Q = nn.Linear(2, 2)
        self.W_K = nn.Linear(2, 2)
        self.W_V = nn.Linear(2, 2)

    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.T
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        return output


model = SelfAttention(2)

x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

output = model(x)
print("Output:")
print(output)
old_weight = model.W_Q.weight.detach().clone()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

target = torch.zeros_like(output)
print("BEFORE")

print("W_Q:")
print(model.W_Q.weight)

print("W_K:")
print(model.W_K.weight)

print("W_V:")
print(model.W_V.weight)


optimizer.zero_grad()

output = model(x)

loss = ((output - target) ** 2).mean()

loss.backward()

optimizer.step()


print("\nAFTER")

print("W_Q:")
print(model.W_Q.weight)

print("W_K:")
print(model.W_K.weight)

print("W_V:")
print(model.W_V.weight)
