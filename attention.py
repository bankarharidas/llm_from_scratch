import torch

# Three token vectors
x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# For now, use the same vectors for Q, K and V
Q = x
K = x
V = x

# 1. Calculate attention scores
scores = Q @ K.T

# 2. Convert scores to attention weights
weights = torch.softmax(scores, dim=-1)

# 3. Use the weights to combine Values
output = weights @ V

print("Scores:")
print(scores)

print("Weights:")
print(weights)

print("Output:")
print(output)
