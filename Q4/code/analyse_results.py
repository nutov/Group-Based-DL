# take specific results folder and make plots for it
import os
from file_manager import SaveFig, LoadData, logger, set_result_dir, DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
from common import RESULT_DIR, TRAIN_FLAG, N_EPOCHS

def analyse_results(path):

    set_result_dir(path)

    # load test and validation accuracy for all k and plot the models accuracies
    k_list = ('5', '10', '50', 'all')
    accuracy_per_k_per_method = [[] for _ in range(len(k_list))]  # list[k][method]
    n_methods = None

    for k in k_list:

        test_accuracy = LoadData(f"test_accuracy_per_epoch_k_{k}")  # list[method][epoch],

        n_methods = len(test_accuracy)
        for i in range(n_methods):
            resulted_accuracy = test_accuracy[i][-1]
            accuracy_per_k_per_method[k_list.index(k)].append(resulted_accuracy)



    # plot
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    markers = ['o', 'v', '^', '<', '>', 's', 'P']
    labels = ['Basic', 'Canonization', 'Symmetrization', 'Sampled Symmetrization', 'Equivariant', 'Augmented']
    accuracy_per_method_per_k = np.array(accuracy_per_k_per_method).T  # shape: (n_methods, len(k_list))

    x = np.arange(len(k_list))
    plt.figure(figsize=(10, 6))
    for i in range(n_methods):
        plt.plot(x, accuracy_per_method_per_k[i],
            marker=markers[i],
            color=colors[i],
            label=labels[i],
            linestyle='-')

    plt.xticks(x, k_list)
    plt.xlabel('k')
    plt.ylabel('Test accuracy')
    plt.title('Model accuracy vs k')
    plt.grid(True)
    plt.legend()
    SaveFig('accuracy_vs_k')
    plt.show()



if __name__ == '__main__':
    r'/home/tuvy/Documents/study/deep_and_groups_hw4/Group-Based-DL/Q4/code/analyse_results.py'
    path = r'/home/tuvy/Documents/study/deep_and_groups_hw4/Group-Based-DL/Q4/code/results/20250811_1447'
    analyse_results(path)