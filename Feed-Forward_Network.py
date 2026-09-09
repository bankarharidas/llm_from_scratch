import torch.nn as nn

ffn = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))

total = 0

for parameter in ffn.parameters():
    print(parameter.shape)
    total += parameter.numel()

print("Total parameters:", total)
