import data
from Q4.code.data import get_data_path
from viz import *

data_n = int(256)
data_dim = int(3)

if __name__ == "__main__":
    absolute_path_to_data = get_data_path()
    cloud_data = data.PointCloudDataset(absolute_path_to_data)
    pcd = cloud_data.show_data(cloud_data.categories[0], 1, num_samples=256)[0]
    plot_pcd(pcd)

    tensor, label = cloud_data.__getitem__(100)
    
    print(f"label: {label}\n",
          f"len: {len(cloud_data)}\n"
          f"tensor: \n {tensor}")
          