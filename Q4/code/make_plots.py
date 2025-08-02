from viz import plot_pcd, plot_pcd_2D
from file_manager import SaveFig
import data
import models
import matplotlib.pyplot as plt
import numpy as np
import torch

# def plot_squared_image_of_all_labels(dataset: data.BaseDataset):
#     """
#     Plot a squared image of all labels_to_numbers in the dataset.
#     :param dataset: torch.utils.data.Dataset object
#     """
#     labels = dataset.labels_to_numbers
#     num_labels = len(labels)
#     num_cols = int(num_labels ** 0.5)
#     num_rows = num_cols
#
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
#     axs = axs.flatten()
#
#     for i, (label_name, label_num) in enumerate(labels.items()):
#         if i >= num_rows * num_cols:
#             break
#         pcd = dataset.filenames_per_label[label_name][0]  # Get the first point cloud for each label TODO: fix it if needed
#         plot_pcd_2D(pcd, fig_num=None)
#         axs[i].set_title(label_name)
#         axs[i].axis('off')
#
#     plt.tight_layout()
#     SaveFig(fig, "squared_image_of_all_labels")
#     plt.show()

def plot_object_and_scores(dataset: data.BaseDataset, model: models.BasePointCloudNet, inx=0):
    """
    Plot two subsplots:
    1) The point cloud object.
    2) The scores for each label. This is a bar plot where the y-axis is the labels_to_numbers and the x-axis is the scores.
    input_data: torch.Tensor         The point cloud data to be plotted.
    scores: torch.Tensor             The scores for each label.
    list_of_labels: list            The list of labels_to_numbers corresponding to the scores.
    """

    input_data, true_label = dataset[inx]
    classifier_output = model(input_data)
    plot_object_and_scores_helper(classifier_output, input_data, true_label, dataset)



def plot_object_and_scores_helper(classifier_output:torch.tensor, input_data:torch.tensor, true_label, dataset):

    true_label_name = dataset.numbers_to_labes[true_label]
    scores = classifier_output.to(device="cpu").detach().numpy()
    scores = np.squeeze(scores)
    input_data = input_data.to(device="cpu").detach().numpy()
    list_of_labels = dataset.numbers_to_labes

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the point cloud object
    plot_pcd_2D(input_data, ax=axs[0])
    axs[0].set_title(f"Point Cloud of {true_label_name.title()}")
    axs[0].axis('equal')
    axs[0].grid(True)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])

    # Plot the scores
    axs[1].barh(list_of_labels, scores)
    axs[1].set_xlabel("Score")
    axs[1].set_ylabel("Labels")

def accuracy_vs_epochs(accuracies: dict, ax=None):
    """
    Plot the accuracy vs epochs for multiple models.
    :param accuracies: dictionary where keys are model names and values are 1D numpy arrays of accuracies for at each epoch
    """
    if ax is None:
        ax = plt.gca()

    model_names = accuracies.keys()
    for model_name in model_names:
        acc = accuracies[model_name]
        n = len(acc)
        x = range(1, n + 1)
        ax.plot(x, acc, label=model_name)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Epochs')
    ax.legend()