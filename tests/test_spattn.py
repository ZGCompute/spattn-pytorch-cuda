import torch
from models.spattn import SpAttn

def test_spattn_forward():
    model = SpAttn(dim=256, heads=4, sparsity=0.5).cuda()
    x = torch.randn(2, 256, 32, 32).cuda()
    y = model(x)
    assert y.shape == x.shape, f"Unexpected output shape: {y.shape}"
