import os
import os.path as osp
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

class PointCloudDataset(Dataset):
    def __init__(self, path_to_data, num_points=256, data_dim = 3, transform=None):
        
        self.base_path = path_to_data
        self.categories = self._get_categories()
        self.category_to_label = self._category_to_label()  # dict
        self.files_per_category = self._get_files_per_category()  # dict(categoryName: fileName)
        self.all_files = self._get_all_files()

        self.num_points = num_points
        self.data_dim = data_dim
        self.transform = transform



    def _get_categories(self) -> list:
        dirs = os.listdir(self.base_path)
        categories = \
            [f for f in dirs if osp.isdir(osp.join(self.base_path, f)) and not f.startswith(".")]
        return categories
    
    def _category_to_label(self) -> dict:
        """returns the {label (number): category} dictionary"""
        return {category: i for i, category in enumerate(self.categories)}

    def _get_files_per_category(self) -> dict:
        # Returns a list of (file_path, category_index)
        return{categ: os.listdir(osp.join(self.base_path, categ))  for categ in self.categories}
    
    def _get_all_files(self) -> list:
        return [(category, f) for category, files in self.files_per_category.items() for f in files]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        category, file = self.all_files[idx]
        label = self.category_to_label[category]
        data_path = osp.join(self.base_path, category, file)
        this_data = np.loadtxt(data_path, delimiter=',')
        this_data = this_data[:self.num_points, :self.data_dim]

        point_cloud = torch.tensor(this_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        return point_cloud, label
    
    def sample_Pcd_per_category(self,category:str,n:int,num_samples:int = 256):
        files = self.files_per_category[category]
        pcd = []
        for idx in np.random.choice(len(files),n,replace=False):
            file_path =  osp.join(self.base_path, category, files[idx])
            pcd.append(np.loadtxt(file_path,delimiter=',')[:num_samples][:,:3])
        return pcd
    
    
