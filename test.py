import numpy  as np
import torch

a = torch.tensor([1, 2, 3, 4, 5])
b = [True, False, True, True, False]

a[b].zero_()

print(a)