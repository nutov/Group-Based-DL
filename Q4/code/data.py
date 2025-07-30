import json
import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
import torch



class base_dataset(Dataset):
    def __init__(self, test_or_train, num_points=256, data_dim=3):
        """
        :param test_or_train: get the values "test" or "train" to load the respective dataset
        :param num_points:  number of points that was specified to be used in the dataset
        :param data_dim:  dimension of data specified to be used in the dataset (there is 6 in total - 3 for point and 3 normal)
        """

        self.base_path = base_dataset._get_data_path()
        self.labels = self._get_labels()  # dictionary {label-name: label-number}
        self.test_or_train = test_or_train  # "test" or "train"
        self.filenames = self._get_filenames()  # list of all filenames

        self.num_points = num_points
        self.data_dim = data_dim

        self.preprocess_success = self.preprocess_data()


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


    def _get_filenames(self):
        file_path = os.path.join(self.base_path, f"modelnet40_{self.test_or_train}.txt")
        return self._read_lines(file_path)


    def preprocess_data(self):
        """
        Preprocess the data.
        Each file such as "airplane_0628.txt" is read to matrix Mdata. We calculate the mean of the self.data_dim first columns to Mmean and the maximum value of them to Mmax.
        Then, we save (Mdata[:self.num_points, :self.data_dim] - Mmean[None, :]) / Mmax[None, :] to the file with the same name and prefix "processed_".
        If the preprocessing is successful, we set return True, otherwise False.
        """
        # check if the data is already preprocessed
        key = f"{self.test_or_train}_data_preprocessed"
        if base_dataset._get_config_value(key):
            return True

        for filename in self.filenames:
            label_name, absolute_path = self._absolute_path(filename)
            input_data = np.loadtxt(absolute_path, delimiter=',', dtype=np.float32)  # shape[N, 6]
            Mmean = np.mean(input_data, axis=0)[:self.data_dim]  # [data_dim]
            input_data = input_data[:, :self.data_dim] - Mmean[None, :]  # [N, data_dim]
            Mmax = np.max(np.abs(input_data), axis=0)[:self.data_dim]  # [data_dim]
            assert np.any(Mmax > 0), f"Max value for {label_name} is zero, cannot normalize data."
            input_data = input_data[:self.num_points, :] / Mmax[None, :]
            processed_path = osp.join(self.base_path, label_name, 'processed_' + filename + '.txt')
            np.savetxt(processed_path, input_data, delimiter=',', fmt='%.6f')

        base_dataset._set_config_value(key, True)
        return True

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
        data_path = base_dataset._get_config_value('data_path')
        if not osp.isdir(data_path):
            raise FileNotFoundError(
                f"The absolute path to the dataset <modelnet40_normal_resampled> should be specified in the config.json in  located at {osp.join(osp.dirname(__file__), 'config.json')}. "
                f"Current value is {data_path}, but the directory does not exist.")
        return data_path


class TrainDataset(base_dataset):
    def __init__(self):
        super().__init__(test_or_train='train')


class TestDataset(base_dataset):
    def __init__(self):
        super().__init__(test_or_train='test')
