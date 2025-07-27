import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import math


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
        norms = torch.norm(x, dim=1)
        _, idx = torch.sort(norms, descending=True, stable=True)

        x = x[idx,:]
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
        x_ = torch.zeros_like(self.linear(x))
        for perm in permutations(range(N)): 
            x_ += self.linear(x[list(perm)])
        return x_/ math.factorial(N)  
        
            

class Sampled_Symmetrization_Net(nn.Module):
    def __init__(self,d = 10,num_samples = 20):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.num_samples = num_samples

    def forward(self, x):
        N,_ = x.size()
        out = torch.zeros_like(self.linear(x))

        for _ in range(self.num_samples):
            perm = torch.randperm(N)#, device=x.device)
            out += self.linear(x[perm,:])
        return out.mean(dim=0)   

        #it = create_permutations_sampled(x,self.num_samples)
        #for perm in it:
        #    out += self.linear(x[perm,:])
        
        #return out / self.num_samples


class Linear_eq_layer(nn.Module):
    def __init__(self, d_in=10, d_hidden=32):
        super().__init__()
        self.w1 = nn.Linear(d_in,d_hidden)
        self.w2 = nn.Linear(d_in,d_hidden)
        
        
    def forward(self, x):  # x is (n, d_in)
        return self.w1(x) + self.w2(torch.unsqueeze(torch.sum(x,dim=0),dim=0))
    

class Linear_eq_Net(nn.Module):
    def __init__(self, d_in=10, d_hidden=32 , d_out = 4):
        super().__init__()
        self.equiv = nn.Sequential(Linear_eq_layer(d_in,d_hidden),
                                   nn.ReLU(),
                                   Linear_eq_layer(d_hidden,d_hidden)
        )

        self.post_pool = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_hidden, d_out))
        
        
    def forward(self, x):  # x is (n, d_in)
        x_eq = self.equiv(x)
        x_pool = x_eq.sum(dim=0)
        return self.post_pool(x_pool)


class AugmentedInvariantNet(nn.Module):
    def __init__(self, d=10, d_hidden=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            #nn.Flatten(),  # input shape (n, d) → (n*d,)
            nn.Linear(d , d_hidden),  
            nn.ReLU(),
            #nn.Linear(d_hidden , d_hidden),
            #nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self, x):
        return self.net(x)