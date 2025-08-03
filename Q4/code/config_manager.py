import os.path as osp
import json

class ConfigManager:
    def __init__(self, config_filename='config.json'):
        self.config_path = osp.join(osp.dirname(__file__), config_filename)

    def get_value(self, key):
        with open(self.config_path) as f:
            thisJson = json.load(f)
        return thisJson[key]

    def set_value(self, key, value):
        with open(self.config_path) as f:
            thisJson = json.load(f)
        thisJson[key] = value
        with open(self.config_path, 'w') as f:
            json.dump(thisJson, f, indent=4)

    def get_data_path(self):
        data_path = self.get_value('data_path')
        if not osp.isdir(data_path):
            raise FileNotFoundError(
                f"The absolute path to the dataset <modelnet40_normal_resampled> should be specified in the config.json in  located at {self.config_path}. "
                f"Current value is {data_path}, but the directory does not exist.")
        return data_path

