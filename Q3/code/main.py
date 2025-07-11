import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
from utils import *
from models import *

def main():
    num_test = 50
    #a = (test_canonization_net,Canonization_Net)
    #b = (test_symmetrization_net,Symmetrization_Net)
    #c = (test_sampled_symmetrization_net,Sampled_Symmetrization_Net)

    #print(f'percent of non invariant canonization {run_test(a,num_tests=num_test)}')
    #print(f'percent of non invariant symmeriztion {run_test(b,num_tests=num_test)}')
    #print(f'percent of non invariant sampled symmeriztion {run_test(c,num_tests=num_test)}')
    n = 10000
    d=50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((n,d)).to(device)
    
    variance_model = AugmentedInvariantNet(d=d).to(device)
    optimizer = torch.optim.Adam(variance_model.parameters(), lr=0.001)
    
    variance_model = train_variance_net(variance_model, optimizer, x,epochs=1000)
    print(f'percent of non invariant sampled symmeriztion {run_test((test_variance_invariance,variance_model),num_tests=num_test)}')

main()