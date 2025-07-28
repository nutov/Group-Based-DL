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


class base_dataset(Dataset):
    def __init__(self, test_or_train, num_points=256, data_dim=3):
        """
        :param test_or_train: get the values "test" or "train" to load the respective dataset
        :param num_points:  number of points that was specified to be used in the dataset
        :param data_dim:  dimension of data specified to be used in the dataset (there is 6 in total - 3 for point and 3 normal)
        """

        self.base_path = get_data_path()
        self.labels = self._get_labels()  # dictionary {label-name: label-number}
        self.filelist_path = self._get_filelist_relative_name()
        self.filenames = self._get_filenames(test_or_train)  # list of all filenames

        self.num_points = num_points
        self.data_dim = data_dim

        self.preprocess_success = False
        self.preprocess_data()


    @staticmethod
    def _read_lines(path):
        if not osp.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines


    def _get_labels(self):
        """returns {label-number: label-name} dictionary
        List of labels is in modelnet40_shape_names.txt
        each line is a label name
        """
        labels_path = osp.join(self.base_path, 'modelnet40_shape_names.txt')
        lines = self._read_lines(labels_path)
        return {label: i for i, label in enumerate(lines)}


    def _get_filelist_relative_name(self) -> str:
        raise NotImplementedError("This method should be implemented in subclasses")


    def _get_filenames(self, files_relative_name: str):
        file_path = os.path.join(self.base_path, files_relative_name)
        return self._read_lines(file_path)


    def preprocess_data(self):
        """
        Preprocess the data.
        Each file such as "airplane_0628.txt" is read to matrix Mdata. We calculate the mean of the self.data_dim first columns to Mmean and the maximum value of them to Mmax.
        Then, we save (Mdata[:self.num_points, :self.data_dim] - Mmean[None, :]) / Mmax[None, :] to the file with the same name and prefix "processed_".
        If the preprocessing is successful, we set return True, otherwise False.
        """
        for filename in self.filenames:
            label_name, absolute_path = self._absolute_path(filename)
            input_data = np.load(absolute_path)


    def _absolute_path(self, filename: str):
        """The filename is composed of category_4d"""
        label_name = filename.split('_')[0]
        absolute_path = osp.join(self.base_path, label_name, filename + '.txt')
        return label_name, absolute_path


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label_name, absolute_path = self._absolute_path(filename)
        label = torch.tensor(self.labels[label_name], dtype=torch.int64)
        input_data = np.loadtxt(absolute_path, delimiter=',')
        input_tensor = torch.from_numpy(input_data).to(torch.float32)

        assert  input_tensor.shape[0] == self.num_points, f"Input data has fewer points than expected: {input_tensor.shape[0]} < {self.num_points}"
        assert input_tensor.shape[1] == self.data_dim, f"Input data has fewer dimensions than expected: {input_tensor.shape[1]} < {self.data_dim}"

        return input_tensor, label

class TrainDataset(base_dataset):
    def _get_filelist_relative_name(self) -> str:
        return 'modelnet40_train.txt'

class TestDataset(base_dataset):
    def _get_filelist_relative_name(self) -> str:
        return 'modelnet40_test.txt'




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



