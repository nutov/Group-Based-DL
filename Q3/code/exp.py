import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
from utils import *

x = torch.randn((10,2))

N,_ = x.size()

#norms = torch.linalg.norm(x,dim=1)
_, indices = torch.sort(x[:,0],descending=True)
print(x)

x = x[indices,:]

#print(indices)
#print(norms)
print(x)
#print(torch.linalg.norm(x,dim=1))