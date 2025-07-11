import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split


class Dataset(Dataset):

    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    

def create_dataset(n:int = 1000 ,d:int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((n,d)).to(device)
    N,d = x.size()
    #this is the variance
    
    y = torch.ones((N,)).to(device)
    return Dataset(x,y)

def create_dataloader(n:int = 1000 ,d:int = 100):
    data = create_dataset(n,d)
    return torch.utils.data.DataLoader(dataset=data ,
                                           batch_size=64 ,
                                           shuffle=True )



def rand_permute(X):
    N,d = X.size()
    perm = torch.randperm(N)
    return  X[perm,:]
