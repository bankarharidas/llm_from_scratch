import torch

x = torch.tensor([10.0, 20.0, 30.0])

network_output = torch.tensor([2.0, 4.0, 6.0])

output = x + network_output

print("Input:", x)
print("Network output:", network_output)
print("Residual output:", output)
