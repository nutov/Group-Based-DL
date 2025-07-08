import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np



def create_permutations_sampled(x:torch.tensor,K:int):
    N,_ = x.size()
    for _ in range(K):
        yield np.random.permutation(N)


def test_canonization_net(CanonizationNet, d=10, n=20, tol=1e-5):
    net = CanonizationNet(d=d)
    x = torch.randn(n, d)
    perm = torch.randperm(n)
    x_perm = x[perm]
    
    y = net(x)
    y_perm = net(x_perm)
    
    # Invariant: output should not change
    return torch.allclose(y, y_perm, atol=tol)


def test_symmetrization_net(SymmetrizationNet, d=10, n=10, tol=1e-5):  # small n for factorial runtime
    net = SymmetrizationNet(d=d)
    x = torch.randn(n, d)
    perm = torch.randperm(n)
    x_perm = x[perm,:]
    
    y = net(x)
    y_perm = net(x_perm)
    
    return torch.allclose(y, y_perm, atol=tol)



def test_sampled_symmetrization_net(SampledSymmetrizationNet, d=10, n=10, num_samples=30, tol=1e-3):
    net = SampledSymmetrizationNet(d=d,num_samples = num_samples)
    x = torch.randn(n, d)
    perm = torch.randperm(n)
    x_perm = x[perm]
    
    y = net(x)
    y_perm = net(x_perm)
    
    return torch.allclose(y, y_perm, atol=tol)


def test_equivariance_equivariant_layer(LinearEquivariantLayer,d_in=10, d_out=4, n=6, tol=1e-5):
    net = LinearEquivariantLayer(d_in, d_out)
    x = torch.randn(n, d_in)
    perm = torch.randperm(n)
    
    x_perm = x[perm]
    
    y = net(x)
    y_perm = net(x_perm)
    
    return torch.allclose(y[perm], y_perm, atol=tol)


def test_deepsets_invariance(DeepSets,d_in=10, n=6, tol=1e-5):
    net = DeepSets(d_in=d_in)
    x = torch.randn(n, d_in)
    perm = torch.randperm(n)
    x_perm = x[perm]

    y = net(x)
    y_perm = net(x_perm)

    return torch.allclose(y, y_perm, atol=tol)


def run_test(test_args:tuple,num_tests = 100):
    res = 0
    test_func,net = test_args
    for _ in range(num_tests):
        if not test_func(net):
            res+=1
    return res 