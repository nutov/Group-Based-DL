import json
import os
import os.path as osp

import numpy as np
from torch.cuda import device
from torch.utils.data import Dataset, TensorDataset, Subset
import torch
import viz
from Q4.code.file_manager import DATA_DIR
from file_manager import logger, TimeIt, SaveData, LoadData

import matplotlib.pyplot as plt

class BaseDataset(Dataset):
    def __init__(self, partition_name, num_points=256, data_dim=3, device=None):
        """
        :param partition_name: get the values "test" or "train" to load the respective dataset
        :param num_points:  number of points that was specified to be used in the dataset
        :param data_dim:  dimension of data specified to be used in the dataset (there is 6 in total - 3 for point and 3 normal)
        """
        logger.info(f"Loading {partition_name} dataset.")
        self.base_path = BaseDataset._get_data_path()
        self.labels_to_numbers, self.numbers_to_labes = self._get_labels_dictionaries()  # dictionary {label-name: label-number}
        self.partition_name = partition_name  # "test", "train", "validation"
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
        file_path = os.path.join(self.base_path, f"{self.partition_name}.txt")
        if not osp.isfile(file_path):
            self.create_data_sets()
        return self._read_lines(file_path)


    def _get_filenames_per_label(self):
        """
        Returns a dictionary where keys are labels names and values are lists of filenames for that label.
        """
        filenames_per_label = {label: [] for label in self.labels_to_numbers.keys()}
        for filename in self.filenames:
            label_name = self._get_label_from_filename(filename)
            filenames_per_label[label_name].append(filename)
        return filenames_per_label

    @TimeIt
    def preprocess_data(self):
        """
        Preprocess the data.
        Each file such as "airplane_0628.txt" is read to matrix Mdata. We calculate the mean of the self.data_dim first columns to Mmean and the maximum value of them to Mmax.
        Then, we save (Mdata[:self.num_points, :self.data_dim] - Mmean[None, :]) / Mmax[None, :] to the file with the same name and prefix "processed_".
        If the preprocessing is successful, we set return True, otherwise False.
        """
        # check if the data is already preprocessed
        key = f"{self.partition_name}_data_preprocessed"
        if BaseDataset._get_config_value(key):
            return True

        logger.info(f"Start Preprocessing {self.partition_name} dataset.")
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

    @staticmethod
    def create_data_sets(p = (0.8, 0.1, 0.1)):
        """
        Create filelists: trainlist.txt ,validationlist.txt, testlist.txt for creating datasets
        p is the proportion of data to be used for train, validation, and test sets.
        """
        logger.info("Creating filelists for datasets")
        base_path = BaseDataset._get_data_path()
        train_file = osp.join(base_path, 'train_list.txt')
        test_file = osp.join(base_path, 'test_list.txt')
        validation_file = osp.join(base_path, 'validation_list.txt')

        filelist = osp.join(base_path, 'filelist.txt')
        lines = BaseDataset._read_lines(filelist)
        train_lines = []
        test_lines = []
        validation_lines = []
        assert len(p) == 3, "Proportions p should be a tuple of 3 values (train, validation, test)"
        assert np.isclose(sum(p), 1.0), "Proportions p should sum to 1.0"

        # put 70% of the data to train, 15% to validation, and 15% to test
        go_to_validation = False
        counter = 0
        n = [0, 0]  # counters of inputs in train, validation, and test sets
        for i, line in enumerate(lines):
            counter += 1
            if n[0] / counter < p[0]:
                train_lines.append(line)
                n[0] += 1
            elif n[1] / counter < p[1]:
                validation_lines.append(line)
                n[1] += 1
            else:
                test_lines.append(line)

        with open(train_file, 'w') as f:
            f.write('\n'.join(train_lines))
        with open(test_file, 'w') as f:
            f.write('\n'.join(test_lines))
        with open(validation_file, 'w') as f:
            f.write('\n'.join(validation_lines))

    def make_subset_for_training(self, k: int):
        indices = []
        for label, files in self.filenames_per_label.items():
            # map filenames to dataset indices
            chosen = [self.filenames.index(fn) for fn in files][:k]
            indices.extend(chosen)
        return Subset(self, indices)


class TrainDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(partition_name='train', **kwargs)

class TestDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(partition_name='test', **kwargs)

class ValidationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(partition_name='validation', **kwargs)


class BaseDatasetOnRam(TensorDataset, BaseDataset):
    """
    Loads all data into RAM and behaves like a TensorDataset.
    Inherits from both BaseDataset and TensorDataset.
    """
    def __init__(self, **kwargs):
        BaseDataset.__init__(self, **kwargs)

        # Check if the data is saved to DATA_DIR
        path_to_data = osp.join(self.base_path, f"BaseDatasetOnRam_{self.partition_name}_data.pkl")
        path_to_labels = osp.join(self.base_path, f"BaseDatasetOnRam_{self.partition_name}_labels.pkl")
        if osp.isfile(path_to_data) and osp.isfile(path_to_labels):
            labels  = np.loadtxt(path_to_labels, delimiter=',', dtype=np.int64)
            L = len(labels)
            saved_data = np.loadtxt(path_to_data, delimiter=',', dtype=np.float32)  # shape (L * num_points, data_dim)
            saved_data = saved_data.reshape((L, -1, self.data_dim))  # reshape to (L, num_points, data_dim)
            labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
            data = torch.tensor(saved_data, dtype=torch.float32, device=self.device)
            TensorDataset.__init__(self, data, labels)
        else:
            data = torch.empty((len(self.filenames), self.num_points, self.data_dim), dtype=torch.float32, device=self.device)
            labels = torch.empty(len(self.filenames), dtype=torch.int64, device=self.device)
            for i in range(len(self.filenames)):
                input_tensor, label = BaseDataset.__getitem__(self, i)
                data[i] = input_tensor[:self.num_points, :self.data_dim]
                labels[i] = label
            TensorDataset.__init__(self, data, labels)
            # Save the data to DATA_DIR
            data_to_save = data.cpu().numpy()  # shape (L, num_points, data_dim)
            data_to_save = np.reshape(data_to_save, (-1, self.data_dim))
            np.savetxt(path_to_data, data_to_save, delimiter=',', fmt='%.8f')
            np.savetxt(path_to_labels, labels.cpu().numpy(), delimiter=',', fmt='%d')


class TrainDatasetOnRam(BaseDatasetOnRam):
    def __init__(self, **kwargs):
        super().__init__(partition_name='train', **kwargs)

class TestDatasetOnRam(BaseDatasetOnRam):
    def __init__(self, **kwargs):
        super().__init__(partition_name='test', **kwargs)

class ValidationDatasetOnRam(BaseDatasetOnRam):
    def __init__(self, **kwargs):
        super().__init__(partition_name='validation', **kwargs)