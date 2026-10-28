import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("Is XPU available?", torch.xpu.is_available())

if torch.xpu.is_available():
    print("GPU Device:", torch.xpu.get_device_name(0))

#


device = torch.device("xpu")
print(f"Using device: {torch.xpu.get_device_name(0)}")

# Tiny model for testing
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)

x = torch.randn(8, 64, device=device)
target = torch.randint(0, 2, (8,), device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Forward + Backward
optimizer.zero_grad()
output = model(x)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"Test step passed! Loss: {loss.item():.4f}")
