import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

x = torch.tensor([1.0, 2.0, 3.0, 4.0])

target = torch.tensor([10.0, 20.0])

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(1000):

    prediction = model(x)

    loss = loss_fn(prediction, target)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print("step: ", step, "loss:", loss.item())

print("Final prediction:", model(x))
