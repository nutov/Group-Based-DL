import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_pcd(pcd, fig_num=None):
    """Plot a 3D point
    @:param pcd: np.array shaped (N, 3) where N is the number of points
    """
    if fig_num is not None:
        plt.figure(fig_num)
    else:
        plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2],s=2)
    # label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def plot_pcd_2D(pcd, fig_num=None):
    """Plot a 2D projection of a point cloud
    @:param pcd: np.array shaped (N, 3) where N is the number of points
    """
    if fig_num is not None:
        plt.figure(fig_num)
    else:
        plt.figure()
    plt.scatter(pcd[:, 0], pcd[:, 1], s=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)