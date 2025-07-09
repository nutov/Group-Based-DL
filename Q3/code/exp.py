import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
from utils import *
from models import *

def main():
    num_test = 50
    a = (test_canonization_net,Canonization_Net)
    b = (test_symmetrization_net,Symmetrization_Net)
    c = (test_sampled_symmetrization_net,Sampled_Symmetrization_Net)

    print(f'percent of non invariant canonization {run_test(a,num_tests=num_test)}')
    print(f'percent of non invariant symmeriztion {run_test(b,num_tests=num_test)}')
    print(f'percent of non invariant sampled symmeriztion {run_test(c,num_tests=num_test)}')
    

#main()


print(test_equivariance(Linear_eq_layer))