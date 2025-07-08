import torch
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
from utils import *
from models import *



a = (test_canonization_net,Canonization_Net)
b = (test_symmetrization_net,Symmetrization_Net)
c = (test_sampled_symmetrization_net,Symmetrization_Net)


