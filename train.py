import torch
import torch.nn as nn

# Model
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

# Input
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Correct answer
target = torch.tensor([10.0, 20.0])

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training
for step in range(1000):

    # Forward
    prediction = model(x)

    # Loss
    loss = loss_fn(prediction, target)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

print("Final prediction:", model(x))
