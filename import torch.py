import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

logits = torch.tensor(
    [[2.0, 5.0, 0.1, 0.2, 0.3], [0.1, 0.2, 4.0, 0.3, 0.5], [0.2, 0.1, 0.3, 5.0, 0.4]]
)

targets = torch.tensor([1, 2, 3])

loss = loss_function(logits, targets)

print("Loss:", loss)
