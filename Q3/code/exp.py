import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np

x = torch.randn((1000,2))

N,_ = x.size()
permutations_ = np.array(list(permutations(np.arange(0,N))))


def create_permutations_sampled(x:torch.tensor,K:int):
    N,_ = x.size()
    return np.array(list(permutations(np.arange(0,N))))[:,np.arange(0,N)][np.random.randint(N,size=K),:]

print(create_permutations_sampled(x,3))

