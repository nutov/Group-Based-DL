import matplotlib.pyplot as plt


def plot_pcd(pcd, ax=None):
    """Plot a 3D point
    @:param pcd: np.array shaped (N, 3) where N is the number of points
    """
    if ax is None:
        ax = plt.gca()
    ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2],s=2)
    # label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def plot_pcd_2D(pcd, ax=None):
    """Plot a 2D projection of a point cloud
    @:param pcd: np.array shaped (N, 3) where N is the number of points
    """
    if ax is None:
        ax = plt.gca()
    ax.scatter(pcd[:, 0], pcd[:, 1], s=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)