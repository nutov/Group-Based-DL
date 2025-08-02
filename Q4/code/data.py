import json
import os
import os.path as osp

import numpy as np
from torch.cuda import device
from torch.utils.data import Dataset
import torch
import viz
from file_manager import logger

import matplotlib.pyplot as plt

class BaseDataset(Dataset):
    def __init__(self, test_or_train, num_points=256, data_dim=3, device=None):
        """
        :param test_or_train: get the values "test" or "train" to load the respective dataset
        :param num_points:  number of points that was specified to be used in the dataset
        :param data_dim:  dimension of data specified to be used in the dataset (there is 6 in total - 3 for point and 3 normal)
        """
        logger.info(f"Loading {test_or_train} dataset.")
        self.base_path = BaseDataset._get_data_path()
        self.labels_to_numbers, self.numbers_to_labes = self._get_labels_dictionaries()  # dictionary {label-name: label-number}
        self.test_or_train = test_or_train  # "test" or "train"
        self.filenames = self._get_filenames()  # list of all filenames
        self.filenames_per_label = self._get_filenames_per_label()  # dictionary {label-name: list of filenames}

        self.num_points = num_points
        self.data_dim = data_dim

        self.preprocess_success = False
        self.preprocess_success = self.preprocess_data()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device



    @staticmethod
    def _read_lines(path):
        if not osp.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines


    def _get_labels_dictionaries(self):
        """returns {label-number: label-name} dictionary
        List of labels_to_numbers is in modelnet40_shape_names.txt
        each line is a label name
        """
        labels_path = osp.join(self.base_path, 'modelnet40_shape_names.txt')
        lines = self._read_lines(labels_path)
        labels_to_numbers = {label: i for i, label in enumerate(lines)}
        return labels_to_numbers, lines


    def _get_filenames(self):
        file_path = os.path.join(self.base_path, f"modelnet40_{self.test_or_train}.txt")
        return self._read_lines(file_path)


    def _get_filenames_per_label(self):
        """
        Returns a dictionary where keys are label names and values are lists of filenames for that label.
        """
        filenames_per_label = {label: [] for label in self.labels_to_numbers.keys()}
        for filename in self.filenames:
            label_name = self._get_label_from_filename(filename)
            filenames_per_label[label_name].append(filename)
        return filenames_per_label


    def preprocess_data(self):
        """
        Preprocess the data.
        Each file such as "airplane_0628.txt" is read to matrix Mdata. We calculate the mean of the self.data_dim first columns to Mmean and the maximum value of them to Mmax.
        Then, we save (Mdata[:self.num_points, :self.data_dim] - Mmean[None, :]) / Mmax[None, :] to the file with the same name and prefix "processed_".
        If the preprocessing is successful, we set return True, otherwise False.
        """
        # check if the data is already preprocessed
        key = f"{self.test_or_train}_data_preprocessed"
        if BaseDataset._get_config_value(key):
            return True

        logger.info(f"Preprocessing {self.test_or_train} dataset.")
        for filename in self.filenames:
            # get data
            label_name, absolute_path = self._absolute_path(filename)
            input_data = np.loadtxt(absolute_path, delimiter=',', dtype=np.float64)[: , :self.data_dim]    # shape (n, 3)

            # calculate center and main axis
            center = np.mean(input_data, axis=0)  # [data_dim]
            input_data -= center[None, :]  # (n, 3)
            scale = np.percentile(np.linalg.norm(input_data, axis=1), 99)
            input_data /= scale
            processed_path = osp.join(self.base_path, label_name, 'processed_' + filename + '.txt')
            np.savetxt(processed_path, input_data, delimiter=',', fmt='%.8f')
            # processed_shortened_path = osp.join(self.base_path, label_name, 'processed_shortened_' + filename + '.txt')
            # np.savetxt(processed_shortened_path, input_data[:self.num_points, :], delimiter=',', fmt='%.8f')

        BaseDataset._set_config_value(key, True)
        return True

    def _absolute_path(self, filename: str):
        """The filename is composed of category_4d"""
        label_name = self._get_label_from_filename(filename)
        if self.preprocess_success:
            absolute_path = osp.join(self.base_path, label_name, "processed_" + filename + '.txt')
        else:
            absolute_path = osp.join(self.base_path, label_name, filename + '.txt')
        return label_name, absolute_path

    @staticmethod
    def _get_label_from_filename(filename: str):
        return '_'.join(filename.split('_')[:-1])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label_name, absolute_path = self._absolute_path(filename)
        label = torch.tensor(self.labels_to_numbers[label_name], dtype=torch.int64)
        input_data = np.loadtxt(absolute_path, delimiter=',')
        input_tensor = torch.tensor(input_data, device=self.device, dtype=torch.float32)
        # assert input_tensor.shape[0] == self.num_points, f"Input data has fewer points than expected: {input_tensor.shape[0]} < {self.num_points}"
        # assert input_tensor.shape[1] == self.data_dim, f"Input data has fewer dimensions than expected: {input_tensor.shape[1]} < {self.data_dim}"

        return input_tensor, label


    @staticmethod
    def _get_config_value(key):
        config_path = osp.join(osp.dirname(__file__), 'config.json')
        with open(config_path) as f:
            thisJson = json.load(f)
        return thisJson[key]

    @staticmethod
    def _set_config_value(key, value):
        config_path = osp.join(osp.dirname(__file__), 'config.json')
        with open(config_path) as f:
            thisJson = json.load(f)
        thisJson[key] = value
        with open(config_path, 'w') as f:
            json.dump(thisJson, f, indent=4)

    @staticmethod
    def _get_data_path():
        data_path = BaseDataset._get_config_value('data_path')
        if not osp.isdir(data_path):
            raise FileNotFoundError(
                f"The absolute path to the dataset <modelnet40_normal_resampled> should be specified in the config.json in  located at {osp.join(osp.dirname(__file__), 'config.json')}. "
                f"Current value is {data_path}, but the directory does not exist.")
        return data_path


    # def show_data(self, idx=None, label=None):
    #     """
    #     Visualize the point cloud data.
    #     :param idx: index of the data to visualize, if None, random index is chosen
    #     :param label: label of the data to visualize, if None, the image is taken from the entire dataset
    #     :return: None
    #     """
    #     if label is not None:
    #         if idx is None:
    #             idx = np.random.randint(0, len(self.filenames_per_label[label]))
    #
    #         filename = self.filenames_per_label[label][idx]
    #     else:
    #         if idx is None:
    #             idx = np.random.randint(0, len(self.filenames))
    #         filename = self.filenames[idx]
    #
    #     label_name, absolute_path = self._absolute_path(filename)
    #     res = np.loadtxt(absolute_path, delimiter=',', dtype=np.float32)  # shape[N, 6]
    #     viz.plot_pcd(res[:, :self.data_dim])
    #     plt.show()



class TrainDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(test_or_train='train', **kwargs)


class TestDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(test_or_train='test', **kwargs)
