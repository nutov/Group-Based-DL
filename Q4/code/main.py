import matplotlib.pyplot as plt
import numpy as np

import data
from viz import *
# from models import *
import os
import torch
import models
from file_manager import SaveFig, logger, SaveData
import utils
from make_plots import plot_object_and_scores

TRAIN_FLAG = False


data_n = int(256)
data_dim = int(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    train_dataset = data.TrainDataset()
    test_dataset = data.TestDataset()

    # plot_pcd(train_dataset[0][0])
    # plt.show()
    # plot_pcd(test_dataset[0][0])
    # plt.show()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    # train LinearEquivariantNet model
    equivariant_model = models.LinearEquivariantNet().to(device=device)

    if TRAIN_FLAG:
        optimizer = torch.optim.Adam(equivariant_model.parameters(), lr=0.001)
        equivariant_model, train_losses, test_losses = utils.train(equivariant_model, optimizer, train_dataloader, test_dataloader, epochs=10)
        plt.figure(1)
        plt.semilogy(np.array(train_losses) + 1e-13, label='Train Loss')
        plt.semilogy(np.array(test_losses) + 1e-13, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        SaveFig("equivariant_model_loss_plot")
        plt.show()
        SaveData(train_losses, "train_losses_equivariant_model")
        SaveData(test_losses, "test_losses_equivariant_model")
        equivariant_model.save()
    else:
        equivariant_model.load()
        equivariant_model.eval()
        # test invariance to permutations
        first_input = train_dataset[0][0]
        for i in range(10):
            utils.test_equivariance_equivariant_layer(equivariant_model, first_input)


    # check results for different inputs
    output_per_label = []
    for label in train_dataset.numbers_to_labes:
        # get index of the first point cloud with this label
        filename = train_dataset.filenames_per_label[label][0]
        inx = train_dataset.filenames.index(filename)
        input_data, true_label = train_dataset[inx]
        plot_object_and_scores(train_dataset, equivariant_model, inx=inx)
        SaveFig(filename)
        plt.close()
        output_data = equivariant_model(input_data).detach().to("cpu").numpy()
        output_per_label.append(output_data)

    # check if the output_data is different for each input data and the Forbinius norm for each pair of outputs
    n = len(output_data)
    distances = np.zeros((n, n))
    for i in range(len(output_per_label)):
        for j in range(i+1, len(output_per_label)):
            distances[i, j] = np.linalg.norm(output_per_label[i] - output_per_label[j])
            distances[j, i] = distances[i, j]

    # print the distance matrix logger and specify the format to be 3 decimal places
    logger.info("Distance matrix between outputs for different labels:")
    for row in distances:
        logger.info(" ".join(f"{val:.3f}" for val in row))






    plot_object_and_scores(train_dataset, equivariant_model)
    SaveFig("equivariant_train_data_example")
    plt.show()

    plot_object_and_scores(test_dataset, equivariant_model)
    SaveFig("equivariant_test_data_example")
    plt.show()




    # # example of ploting a parabula
    # x = np.linspace(-1, 1, 100)
    # y = x ** 2
    # plt.plot(x, y)
    # plt.show()



    # # data preprocessing
    # # preprocess_data()  #TODO
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # # Read data
    # absolute_path_to_data = get_data_path()
    # cloud_data = data.PointCloudDataset(absolute_path_to_data)
    # pcd = cloud_data.show_data(cloud_data.categories[0], 1, num_samples=256)[0]
    # plot_pcd(pcd)
    #
    # num_test = 100
    #
    # # network optimizers
    # variance_model = AugmentedInvariantNet(d=data_n * data_n).to(device)
    # optimizer = torch.optim.Adam(variance_model.parameters(), lr=0.005)
    # variance_model = train_model(variance_model, optimizer, epochs=100)
    #
    # # traning_set_size = 10
    # #
    # # # canonization
    # # net_canon = Canonization_Net(d_in = data_n * data_dim)
    # # optimizer = torch.optim.Adam(net_canon.parameters(), lr=0.005)
    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)
    # # net_canon = train_variance_net(net_canon, optimizer, x,epochs=100,sched=scheduler)
    #
    # # # symetrization
    # # # samples 10 permutations from S_n
    # #
    # #
    # # net_canon
    # # a = (test_canonization_net,Canonization_Net)
    # # b = (test_symmetrization_net,Symmetrization_Net)
    # # c = (test_sampled_symmetrization_net,Sampled_Symmetrization_Net)
    # #
    # # print(f'percent of non invariant canonization {run_test(a,num_tests=num_test)}')
    # # print(f'percent of non invariant symmeriztion {run_test(b,num_tests=num_test)}')
    # # print(f'percent of non invariant sampled symmeriztion {run_test(c,num_tests=num_test)}')
    # # n = 500
    # # d=50
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # x = torch.randn((n,d)).to(device)
    #
    #
    #
    #
    #
    # print(f'percent of non equivariant trained model with augmentations: {run_test((test_variance_invariance,variance_model),num_tests=num_test)}')




if __name__ == '__main__':
    main()