import data
import models
import utils
import os
import os.path as osp
import re

def get_models_list(results_path):
    """
    Get a list of all models in the results/data path.
    A model name is of the format: ModelNmae_number.pth
    The result is a dictionary with model names as keys and list of model files as values.
    :param results_path: path to the results directory
    :return: list of model names
    """

    data_path = osp.join(results_path, 'data')
    for dir_name in os.listdir(data_path):
        model_name = dir_name.split('_')[0]  # Extract model name from directory name




    return [f for f in os.listdir(results_path) if f.endswith('.pth')]

# def calculate_model_metrics

if __name__ == '__main__':
    results_path = r"/Q4/code/results/20250803_2143/data"
    models_list = get_models_list(results_path)
    print(models_list)