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


def test_symmetrization_net(SymmetrizationNet, d=3, n=5, tol=1e-5):  # small n for factorial runtime
    net = SymmetrizationNet(d=d)
    x = torch.randn(n, d)
    perm = torch.randperm(n)
    x_perm = x[perm,:]
    
    y = net(x)
    y_perm = net(x_perm)
    
    return torch.allclose(y, y_perm, atol=tol)



def test_sampled_symmetrization_net(SampledSymmetrizationNet, d=5, n=10, num_samples=750, tol=5e-3):
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


def test_equivariance(Linear_eq_layer,d_in=10, n=10, tol=1e-5):
    net = Linear_eq_layer(d_in=d_in)
    x = torch.randn(n, d_in)
    perm = torch.randperm(n)
    x_perm = x[perm]

    y = net(x)[perm]
    y_perm = net(x_perm)

    return torch.allclose(y, y_perm, atol=tol)

def test_invariant_net(model_class, d_in=10, n=6, tol=1e-5):
    net = model_class(d_in=d_in)
    x = torch.randn(n, d_in)
    x_perm = x[torch.randperm(n)]
    y = net(x)
    y_perm = net(x_perm)
    return torch.allclose(y, y_perm, atol=tol)


def run_test(test_args:tuple,num_tests = 100):
    res = 0
    test_func,net = test_args
    for _ in range(num_tests):
        if not test_func(net):
            res+=1
    return res/num_tests


def compute_variance_target(x):
    return torch.unsqueeze(x.var(dim=1, unbiased=True),dim=1)

def custom_loss(outputs, targets):
    prec = 1 / torch.var(outputs)
    not_mse = 0.5 * prec * torch.mean(torch.pow(outputs - targets, 2))
    return not_mse - 0.5 * torch.log(prec)

def train_variance_net(model, optimizer, x, epochs=100, augments_per_epoch=250,sched = None,verbose = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x.to(device)
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(augments_per_epoch):
            perm = torch.randperm(x.size(0),device=device)
            x_aug = x[perm]                          # permute rows
            y_true = compute_variance_target(x_aug).to(device)  # shape: (n,1)
            y_pred = model(x_aug).to(device)
            loss = F.mse_loss(y_pred, y_true)
            #loss = custom_loss(y_pred, y_true)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        lr = None
        if sched is not None:
            sched.step()
            lr = sched.get_last_lr()
        if verbose:
            if epoch % 10 == 0:
                if lr is None:
                    print(f"[Epoch {epoch}] Loss: {total_loss / augments_per_epoch:.6f}")
                else:
                    print(f"[Epoch {epoch}] Loss: {total_loss / augments_per_epoch:.6f},lr:{lr}")

    return model





def test_variance_invariance(model,d=50, tol=1e-1, num_tests=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((100,d)).to(device)
    model.eval()
    with torch.no_grad():
        y_ref = model(x)
        for _ in range(num_tests):
            perm = torch.randperm(x.size(0))
            y_alt = model(x[perm])
            if not torch.allclose(y_ref[perm], y_alt, atol=tol):
                return False
    return True