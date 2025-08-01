from viz import plot_pcd, plot_pcd_2D
from file_manager import SaveFig
import data
import matplotlib.pyplot as plt

def plot_squared_image_of_all_labels(train_dataset: data.TrainDataset):
    """
    Plot a squared image of all labels in the dataset.
    :param train_dataset: torch.utils.data.Dataset object
    """
    labels = train_dataset.labels
    num_labels = len(labels)
    num_cols = int(num_labels ** 0.5)
    num_rows = num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axs = axs.flatten()

    for i, (label_name, label_num) in enumerate(labels.items()):
        if i >= num_rows * num_cols:
            break
        pcd = train_dataset.filenames_per_label[label_name][0]  # Get the first point cloud for each label
        plot_pcd_2D(pcd, fig_num=None)
        axs[i].set_title(label_name)
        axs[i].axis('off')

    plt.tight_layout()
    SaveFig(fig, "squared_image_of_all_labels")
    plt.show()

    
