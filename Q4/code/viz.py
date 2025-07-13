# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


def plot_pcd(pcd):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2],s=2)
    # label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Random Point Cloud")
    # display:
    plt.show()