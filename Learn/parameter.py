import torch
import torch.nn as nn

model = nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[2.0]])
target = torch.tensor([[10.0]])


print("BEFORE")
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())


# Forward
prediction = model(x)

# Loss
loss = (prediction - target) ** 2

# Backward
loss.backward()


print("\nGRADIENT")
print("Weight gradient:", model.weight.grad.item())
print("Bias gradient:", model.bias.grad.item())


# Update
optimizer.step()


print("\nAFTER")
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
