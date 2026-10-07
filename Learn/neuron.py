import torch

x = torch.tensor(3.0)

weight = torch.tensor(2.0, requires_grad=True)
bias = torch.tensor(4.0, requires_grad=True)

prediction = x * weight + bias

print("Prediction:", prediction)
