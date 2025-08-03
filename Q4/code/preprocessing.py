import os.path as osp
import numpy as np
from file_manager import logger
from config_manager import ConfigManager


def preprocess_data(base_path, filenames, data_dim, num_points):
    """
    Preprocess the data.
    Each file such as "airplane_0628.txt" is read to matrix Mdata. We calculate the mean of the data_dim first columns to Mmean and the maximum value of them to Mmax.
    Then, we save (Mdata[:num_points, :data_dim] - Mmean[None, :]) / Mmax[None, :] to the file with the same name and prefix "processed_".
    If the preprocessing is successful, we set return True, otherwise False.
    """
    config = ConfigManager()

    logger.info(f"Start Preprocessing.")
    base_path = config.get_data_path()
    for filename in filenames:
        label_name = '_'.join(filename.split('_')[:-1])
        absolute_path = osp.join(base_path, label_name, filename + '.txt')
        input_data = np.loadtxt(absolute_path, delimiter=',', dtype=np.float64)[:, :data_dim]
        center = np.mean(input_data, axis=0)
        input_data -= center[None, :]
        scale = np.percentile(np.linalg.norm(input_data, axis=1), 99)
        input_data /= scale
        processed_path = osp.join(base_path, label_name, 'processed_' + filename + '.txt')
        np.savetxt(processed_path, input_data, delimiter=',', fmt='%.8f')
    return True

def rotation_canonization()
    pass  # TODO maybe later
