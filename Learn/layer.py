from typing import Required

import torch

x = torch.tensor([3.0])

weight = torch.tensor(2.0, requires_grad=True)
bais = torch.tensor(1.0, requires_grad=True)

prediction = weight * x + bais

print(prediction)
