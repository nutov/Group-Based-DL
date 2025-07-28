import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
from utils import *
from models import *

def main():
    
    num_test = 50
    """
    a = (test_canonization_net,Canonization_Net)
    print(f'percent of non invariant canonization {run_test(a,num_tests=num_test)}')
    b = (test_symmetrization_net,Symmetrization_Net)
    c = (test_sampled_symmetrization_net,Sampled_Symmetrization_Net)

    print(f'percent of non invariant symmeriztion {run_test(b,num_tests=num_test)}')
    print(f'percent of non invariant sampled symmeriztion {run_test(c,num_tests=num_test)}')
    d = (test_invariant_net,Linear_eq_Net)
    print(f'percent of non invariant eq_layers net {run_test(d,num_tests=num_test)}')
    """
#---------------------------------------------------------------------------------------------------------------------    
    n = 50
    d=5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((n,d)).to(device)
    
    variance_model = AugmentedInvariantNet(d=d*n).to(device)
    optimizer = torch.optim.Adam(variance_model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)

    
    variance_model = train_variance_net(variance_model, optimizer, x,epochs=250,sched=scheduler)
    print(f'percent of non equivariant trained model with augmentations: {run_test((test_variance_invariance,variance_model),num_tests=num_test)}')

main()