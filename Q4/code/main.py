import data
from Q4.code.data import get_data_path
from viz import *
from models import *
data_n = int(256)
data_dim = int(3)

def main():

    # data preprocessing
    # preprocess_data()  #TODO
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # You should specify the absolute path to the data in config.json!!!
    absolute_path_to_data = get_data_path()
    cloud_data = data.PointCloudDataset(absolute_path_to_data)
    pcd = cloud_data.sample_Pcd_per_category(cloud_data.categories[0],1,num_samples=256)[0]
    plot_pcd(pcd)


    num_test = 100

    # network optimizers
    variance_model = AugmentedInvariantNet(d=data_n * data_n).to(device)
    optimizer = torch.optim.Adam(variance_model.parameters(), lr=0.005)
    variance_model = train_model(variance_model, optimizer, epochs=100)

    # traning_set_size = 10
    #
    # # canonization
    # net_canon = Canonization_Net(d_in = data_n * data_dim)
    # optimizer = torch.optim.Adam(net_canon.parameters(), lr=0.005)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)
    # net_canon = train_variance_net(net_canon, optimizer, x,epochs=100,sched=scheduler)

    # # symetrization
    # # samples 10 permutations from S_n
    #
    #
    # net_canon
    # a = (test_canonization_net,Canonization_Net)
    # b = (test_symmetrization_net,Symmetrization_Net)
    # c = (test_sampled_symmetrization_net,Sampled_Symmetrization_Net)
    #
    # print(f'percent of non invariant canonization {run_test(a,num_tests=num_test)}')
    # print(f'percent of non invariant symmeriztion {run_test(b,num_tests=num_test)}')
    # print(f'percent of non invariant sampled symmeriztion {run_test(c,num_tests=num_test)}')
    # n = 500
    # d=50
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x = torch.randn((n,d)).to(device)
    


    
    
    print(f'percent of non equivariant trained model with augmentations: {run_test((test_variance_invariance,variance_model),num_tests=num_test)}')




if __name__ == '__main__':
    main()