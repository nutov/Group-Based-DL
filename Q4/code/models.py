import torch
import torch.nn.functional as F
import math
from sympy.polys.rootisolation import dup_outer_refine_real_root
from torch import nn
from Q4.code.utils import *


class BasePointCloudNet(nn.Module):
    def __init__(self, n_in=256, d_in=3, d_out=40):

        super().__init__()
        self.n = n_in  # Number of points in the point cloud
        self.dim = d_in  # Dimension of each point in the point cloud
        self.d_in = d_in * n_in  # Input dimension after flattening (n * d)
        self.n_classes = d_out  # Number of output classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_in * n_in, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=d_out)
        )

    def process_input(self, x):
        # Default: no permutation handling
        return x

    def forward(self, x):
        x = x.to(self.device)
        x = self.process_input(x)
        x = self.flatten(x)
        return self.mlp(x)


class CanonicalizationNet(BasePointCloudNet):
    def process_input(self, x):
        # x shape: [n, d]
        # Canonicalize by sorting norms
        norms = torch.norm(x, dim=1)  # The probability that two vectors to have the same norms is assuming continues variables.
        _, indices = torch.sort(norms, descending=True, stable=True)
        return x[indices]


class SymmetrizationNet(BasePointCloudNet):

    """In this case, only sampled symmetrization is computationally feasible, since the number of permutations is n!.
    We implemented a naive symmetrization that sums over all permutations.
    But it cannot run in practice.
    """

    def forward(self, x):
        x = x.to(self.device)
        res = torch.zeros((self.n_classes), device=self.device)
        for perm in torch.permutations(torch.arange(self.n)):
            res += self.mlp(self.flatten(x[list(perm), :]))
        return res / torch.tensor(math.factorial(self.n), dtype=x.dtype, device=self.device)

class SampledSymmetrizationNet(BasePointCloudNet):

    def __init__(self, n_in=256, d_in=3, d_out=40, num_samples=20):
        super().__init__(n_in=n_in, d_in=d_in, d_out=d_out)
        self.num_samples = num_samples

    def 



class DeepSetsNet(BasePointCloudNet):
    def __init__(self, n_in=256, d_in=3, d_out=40):
        super().__init__(n_in=n_in, d_in=d_in, d_out=d_out)
    def process_input(self, x):
        # Permutation invariant: sum over elements
        return x.sum(dim=0)


class Canonization_Net(nn.Module):
    def __init__(self,d_in = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x: torch.tensor):
        """
        X - R^(Nxd) 
        canonize by sorting w.r.t norms of the elements in the dataset , 
        this is permutation invariant  
        """
        norms = torch.norm(x, dim=0)
        _, idx = torch.sort(norms, descending=True, stable=True)

        x = x[idx]
        x = self.flatten(x)

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
        elemnts = [k for k in range(N)]
        for perm in permutations(elemnts):
            x_ += self.linear(x[perm,:])
        return x_
        
            

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
        x_ = torch.zeros_like(self.linear(x))
        it = create_permutations_sampled(x,self.num_samples)
        for perm in it:
            x_ += self.linear(x[perm,:])
        
        return x_ / self.num_samples


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
        self.net = nn.Sequential(
            #nn.Flatten(),  # input shape (n, d) → (n*d,)
            nn.Linear(d , d_hidden),  # assuming n = 10
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self, x):
        #x = x.to(self.device)
        #self.net.to(self.device)
        return self.net(x)