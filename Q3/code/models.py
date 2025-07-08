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
        _, indices = torch.sort(x[:,0],descending=True)

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
        x_ = torch.zeros((4,1))
        elemnts = [k for k in range(N)]
        for perm in permutations(elemnts):
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
        N,_ = x.size()
        x_ = torch.zeros((4,1))
        it = create_permutations_sampled(x,N)
        for perm in it:
            x_ += self.linear(x[perm,:])
        
        return x_


class Linear_eq_Net(nn.Module):
    def __init__(self, d_in=10, d_hidden=32, d_out=4, pooling='sum'):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.rho = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )
        self.pooling = pooling

    def forward(self, x):  # x is (n, d_in)
        x_phi = self.phi(x)  # Apply phi to each element
        if self.pooling == 'sum':
            pooled = x_phi.sum(dim=0)
        elif self.pooling == 'mean':
            pooled = x_phi.mean(dim=0)
        elif self.pooling == 'max':
            pooled, _ = x_phi.max(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.rho(pooled)

