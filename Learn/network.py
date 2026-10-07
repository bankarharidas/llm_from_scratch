import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

print(model)
for name, parameter in model.named_parameters():
    print(name, parameter.shape)


x = torch.tensor([1.0, 2.0, 3.0, 4.0])

output = model(x)

print("Output:", output)
