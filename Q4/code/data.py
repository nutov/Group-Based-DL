import json
import os
import os.path as osp
from os import path as osp

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

def get_data_path():
    config_path = osp.join(osp.dirname(__file__), 'config.json')
    example_path = osp.join(osp.dirname(__file__), 'config.example.json')
    if osp.exists(config_path):
        with open(config_path) as f:
            return json.load(f)["data_path"]
    else:
        with open(example_path) as f:
            return json.load(f)["data_path"]

data_dir = get_data_path()


class modelnet_dataset(Dataset):
    def __init__(self, test_or_train, num_points=256, data_dim=3):
        """
        :param test_or_train:
        :param num_points:
        :param data_dim:
        """
        self.base_path = get_data_path()
        self.relative_datapath_list = relative_datapath_list
        self.num_points = num_points
        self.data_dim = data_dim
        self.test_data_list = self._get_test_data_list()
        pass

    def _filename_composition(self, filename: str):
        """The filename is composed of category_4d"""
        category = filename.split('_')[0]
        absolute_path = osp.join(data_dir, category, filename)
        return category, absolute_path

    def _get_test_data_list(self):
        # read the modelnet40_test/train.txt file
        test_file_path = osp.join(self.base_path, self.relative_datapath_list)
        with open(test_file_path, 'r') as f:
            test_data = [line.strip() for line in f.readlines()]

        return test_data

    def __len__(self):
        return len(self.test_data_list)

    def __getitem__(self, idx):
        filename = self.test_data_list[idx]
        category, absolute_path = self._filename_composition(filename)
        data = np.loadtxt(absolute_path, delimiter=',')
        data = data[:self.num_points, :self.data_dim]




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
        #  TODO: change to use modelnet40_test.txt

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
        this_data = this_data[:self.num_points, :self.data_dim].reshape(-1)

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



