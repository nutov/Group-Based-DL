import torch
import torch.nn.functional as F
from torch import nn
from utils import *


class Canonization_Net(nn.Module):
    def __init__(self,d = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        """
        X - R^(Nxd) 
        canonize by sorting w.r.t norms of the elements in the dataset , 
        this is permutation invariant  
        """
        norms = torch.linalg.norm(x,dim=0)
        _, indices = torch.sort(norms)
        x = x[indices,:]
        return self.linear(x)



class Symmetrization_Net(nn.Module):
    def __init__(self,d = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        N,_ = x.size()
        permutations = torch.combinations(torch.arange(0,N),r=N,with_replacement=False)
        x_ = torch.zeros_like(x)
        for perm in permutations:
            x_ += self.linear(x[perm,:])
        return x_
        
            



class Sampled_Symmetrization_Net(nn.Module):
    def __init__(self,d = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        pass



class Linear_eq_Net(nn.Module):
    def __init__(self,d = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        pass



