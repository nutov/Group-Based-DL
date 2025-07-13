from data import *
from viz import *

def main():
    data = DataSet()
    pcd = data.sample_Pcd_per_category(data.categories[0],1,num_samples=1024)[0]
    plot_pcd(pcd)

if __name__ == '__main__':
    main()