import torch.nn as nn
import torch

# a = torch.arange(0, 32)
# b = torch.stack([a, a]).reshape(2, 2, 4, 4)
b = torch.arange(0, 64).reshape(2, 2, 4, 4)
ln = nn.LayerNorm([4, 4], elementwise_affine=False)
print(b)
print('*'*8)
print(ln(b.float()))



