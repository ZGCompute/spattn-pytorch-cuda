import torch
from models.spattn import SpAttn

x = torch.randn(1, 256, 32, 32).cuda()
model = SpAttn(dim=256, heads=4).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for i in range(10):
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {i}: loss = {loss.item():.4f}")
