import torch
import torch.nn as nn

x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])

layer_norm = nn.LayerNorm(4)

output = layer_norm(x)

print("Input:")
print(x)

print("Output:")
print(output)

print("Gamma:")
print(layer_norm.weight)

print("Beta:")
print(layer_norm.bias)
