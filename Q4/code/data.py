import os 
import numpy as np
from pathlib import Path
# Using os.listdir() and os.path.isdir()

class DataSet():
    def __init__(self):
        self._base_path = Path(os.getcwd()).joinpath(Path('Q4/modelnet40_normal_resampled'))
        self.categories = self._get_categories()
        self.txt_files = self._get_files_per_category()


    def _get_categories(self):
        #[f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        dirs = os.listdir(self._base_path)
        folders = [f for f in dirs if os.path.isdir(self._base_path.joinpath(f)) and not f.startswith(".")]
        return folders

    def _get_files_per_category(self):
        return {categ:os.listdir(self._base_path.joinpath(f'{categ}'))  for categ in self.categories}
    
    def sample_Pcd_per_category(self,category:str,n:int,num_samples:int = 256):
        files = self.txt_files[category]
        pcd = []
        for idx in np.random.choice(len(files),n,replace=False):
            file_path =  self._base_path.joinpath(category).joinpath(files[idx])
            pcd.append(np.loadtxt(file_path,delimiter=',')[:num_samples][:,:3])
        return pcd

        



