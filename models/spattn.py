### `models/spattn.py`

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

spattn_cuda = load(name="spattn_cuda", sources=[
    "cuda_kernels/spattn_kernel.cu"
], verbose=True)

class SpAttn(nn.Module):
    def __init__(self, dim, heads=4, sparsity=0.5):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.sparsity = sparsity
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x.permute(0, 2, 3, 1)).view(B, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        out = spattn_cuda.forward(q.contiguous(), k.contiguous(), v.contiguous(), self.sparsity)
        return self.out(out.view(B, H, W, C)).permute(0, 3, 1, 2)
