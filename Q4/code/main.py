import os

import matplotlib

from common import N_EPOCHS

if os.name == 'nt':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
import numpy as np

import data
from viz import *
import torch
import models
from file_manager import SaveFig, logger
import utils



data_n = int(256)
data_dim = int(3)

def main():

    train_dataset = data.TrainDatasetOnRam()
    test_dataset = data.TestDatasetOnRam()
    validation_dataset = data.ValidationDatasetOnRam()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(test_dataset), shuffle=True)

    for k in (5, 10, 50, -1):

        logger.info(f"\n\nk =  {k}   start \n\n")

        model_list = [
            models.BasePointCloudNet(),
            models.CanonizationNet(),
            models.SymmetrizationNet(),
            models.SampledSymmetrizationNet(),
            models.LinearEquivariantNet(),
            models.AugmentedNet()
        ]

        optimizer_list = [torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4) for model in model_list]
        if k > 0:
            training_subset = train_dataset.make_subset_for_training(k=k)
            training_subset_loader = torch.utils.data.DataLoader(training_subset, batch_size=len(training_subset), shuffle=True)
        else:
            training_subset_loader = train_dataloader

        train_acc, train_loss, test_acc, test_loss = utils.train(
            model_list, optimizer_list, training_subset_loader, test_loader=test_dataloader, validation_loader=validation_dataloader, epochs=N_EPOCHS)


        # plot results
        if k == -1:
            k = 'all'
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
        markers = ['o', 'v', '^', '<', '>', 's', 'P']
        labels = ['Basic', 'Canonization', 'Symmetrization', 'Sampled Symmetrization', 'Equivariant', 'Augmented']
        plt.figure(1)
        # Train Loss (markers only, hollow)
        for i in range(len(model_list)):
            plt.semilogy(
                np.array(train_loss[i]) + 1e-13,
                label=labels[i],
                color=colors[i],
                marker=markers[i],
                linestyle='--',
                markerfacecolor='none',
                markeredgecolor=colors[i],
                markevery=10,
                )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Train Loss per Epoch  k={k}')
        plt.legend()
        plt.grid(True)
        SaveFig(f"k_{k}_equivariant_model_train_loss")
        # plt.show()
        plt.close()

        plt.figure(2)
        # Test Loss (markers only, filled)
        for i in range(len(model_list)):
            plt.semilogy(
                np.array(test_loss[i]) + 1e-13,
                label=labels[i],
                color=colors[i],
                marker=markers[i],
                linestyle='-',
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                markevery=10,
                )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Test Loss per Epoch  k={k}')
        plt.legend()
        plt.grid(True)
        SaveFig(f"k_{k}_equivariant_model_test_loss")
        # plt.show()
        plt.close()

        plt.figure(3)
        # Train Accuracy (markers only, hollow)
        for i in range(len(model_list)):
            plt.plot(
                train_acc[i],
                label=labels[i],
                color=colors[i],
                marker=markers[i],
                linestyle='--',
                markerfacecolor='none',
                markeredgecolor=colors[i],
                markevery=10,
            )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Train Accuracy per Epoch  k={k}')
        plt.legend()
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
        SaveFig(f"k_{k}_equivariant_model_train_accuracy")
        plt.show()

        plt.figure(4)
        # Test Accuracy (markers only, filled)
        for i in range(len(model_list)):
            plt.plot(
                test_acc[i],
                label=labels[i],
                color=colors[i],
                marker=markers[i],
                linestyle='-',
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                markevery=10,
            )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Test Accuracy per Epoch  k={k}')
        plt.legend()
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
        SaveFig(f"k_{k}_equivariant_model_test_accuracy")
        plt.show()
        plt.close()


    # else:
    #     equivariant_model.load(r'/home/tuvy/Documents/study/deep_and_groups_hw4/Group-Based-DL/Q4/code/results_1/data/LinearEquivariantNet100.pth')
    #     # test invariance to permutations
    #     # first_input = train_dataset[0][0]
    #     # y = equivariant_model(first_input)
    #     # for i in range(10):
    #     #     utils.test_equivariance_equivariant_layer(equivariant_model, first_input)
    #
    # # calc accuracy on train and test datasets
    # utils.calculate_accuracy_and_loss(equivariant_model, train_dataloader)
    # utils.calculate_accuracy_and_loss(equivariant_model, test_dataloader)
    #
    # # check results for different inputs
    # output_per_label = []
    # for label in train_dataset.numbers_to_labes:
    #     # get index of the first point cloud with this label
    #     filename = train_dataset.filenames_per_label[label][0]
    #     inx = train_dataset.filenames.index(filename)
    #     input_data, true_label = train_dataset[inx]
    #     plot_object_and_scores(train_dataset, equivariant_model, inx=inx)
    #     SaveFig(filename)
    #     plt.close()
    #     output_data = equivariant_model(input_data).detach().to("cpu").numpy()
    #     output_per_label.append(output_data)
    #
    # # check if the output_data is different for each input data and the Forbinius norm for each pair of outputs
    # logger.info("Distance matrix between outputs for different labels:")
    # for i in range(len(output_per_label)):
    #     for j in range(i+1, len(output_per_label)):
    #         logger(f"{i}, {j}: {np.linalg.norm(output_per_label[i] - output_per_label[j])}")
    #
    #
    #
    # plot_object_and_scores(train_dataset, equivariant_model)
    # SaveFig("equivariant_train_data_example")
    # plt.show()
    #
    # plot_object_and_scores(test_dataset, equivariant_model)
    # SaveFig("equivariant_test_data_example")
    # plt.show()




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