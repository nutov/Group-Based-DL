import torch
from torch import nn
import numpy as np
from file_manager import DATA_DIR, logger
import os.path as osp

class BasePointCloudNet(nn.Module):

    # D_HIDDEN_1, D_HIDDEN_2
    D_HIDDEN_1 = 64
    D_HIDDEN_2 = 128

    def __init__(self, n_in=256, d_in=3, d_out=40):

        super().__init__()
        self.n = n_in  # Number of points in the point cloud
        self.dim = d_in  # Dimension of each point in the point cloud
        self.d_in = d_in * n_in  # Input dimension after flattening (n * d)
        self.n_classes = d_out  # Number of output classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_in * n_in, out_features=self.D_HIDDEN_1),
            nn.ReLU(),
            nn.Linear(in_features=self.D_HIDDEN_1, out_features=self.D_HIDDEN_2),
            nn.ReLU(),
            nn.Linear(in_features=self.D_HIDDEN_2, out_features=d_out)
        )

    def process_input(self, x):
        # Default: no permutation handling
        return x

    def forward(self, x):
        x = self.process_input(x)
        x = self.flatten(x)
        return self.mlp(x)

    def save(self, path=None):
        """Save model by it's name"""
        if path is not None:
            model_name = self.__class__.__name__
            path = osp.join(DATA_DIR, f"{model_name}.pth")
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path=None):
        """Load model from by it's name"""
        if path is not None:
            model_name = self.__class__.__name__
            path = osp.join(DATA_DIR, f"{model_name}.pth")
        if not osp.isfile(path):
            raise RuntimeError(f"Model file does not exist: {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Model loaded from {path}")


class CanonicalizationNet(BasePointCloudNet):
    def process_input(self, x):
        # x shape: [n, d]
        # Canonicalize by sorting norms
        norms = torch.norm(x, dim=1)  # The probability that two vectors to have the same norms is assuming continues variables.
        _, indices = torch.sort(norms, descending=True, stable=True)
        return x[indices]


class SymmetrizationNet(BasePointCloudNet):

    """In this case, only sampled symmetrization is computationally feasible, since the number of permutations is n!.
    We implemented a naive symmetrization that sums over all permutations.
    But it cannot run in practice.
    """

    def forward(self, x):
        res = torch.zeros((self.n_classes), device=self.device)
        for perm in torch.permutations(torch.arange(self.n)):
            res += self.mlp(self.flatten(x[list(perm), :]))
        return res / np.prod(range(1, self.n+1))

class SampledSymmetrizationNet(BasePointCloudNet):

    def forward(self, x):
        n_samples = 20
        res = torch.zeros((self.n_classes), device=self.device)
        for i in range(n_samples):
            perm = torch.randperm(self.n)
            res += self.mlp(self.flatten(x[perm, :]))

        return res / n_samples


class LinearEquivariantLayer(BasePointCloudNet):
    def __init__(self, d_in, d_out):
        super(BasePointCloudNet, self).__init__()
        # input dimension: [n_in, d_in]
        # output dimension: [n_in, d_out]
        self.w1 = nn.Linear(d_in, d_out)
        self.w2 = nn.Linear(d_in, d_out)

    def forward(self, x):
        # x: (n, 3)
        return self.w1(x) + self.w2(torch.sum(x, dim=-2, keepdim=True))  # output: (n, d_out)


class LinearEquivariantNet(BasePointCloudNet):
    def __init__(self, n_in=256, d_in=3, d_out=40):
        super().__init__(n_in, d_in, d_out)
        self.equivariant_mpl = nn.Sequential(
            LinearEquivariantLayer(d_in=d_in, d_out=self.D_HIDDEN_1),
            nn.ReLU(),
            LinearEquivariantLayer(d_in=self.D_HIDDEN_1, d_out=self.D_HIDDEN_2),
            nn.ReLU(),
        )
        self.invariant_mpl = nn.Sequential(
            nn.Linear(in_features=self.D_HIDDEN_2, out_features=self.n_classes)
        )

    def forward(self, x):
        res = self.equivariant_mpl(x)  # res: (n, D_HIDDEN_2)
        return self.invariant_mpl(res.mean(dim=-2))

# TODO: make AugmentedInvariantNet
# class Canonization_Net(nn.Module):
#     def __init__(self,d_in = 10):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Sequential(
#             nn.Linear(d_in, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4)
#         )
#
#     def forward(self, x: torch.tensor):
#         """
#         X - R^(Nxd)
#         canonize by sorting w.r.t norms of the elements in the dataset ,
#         this is permutation invariant
#         """
#         norms = torch.norm(x, dim=0)
#         _, idx = torch.sort(norms, descending=True, stable=True)
#
#         x = x[idx]
#         x = self.flatten(x)
#
#         return self.linear(x)
#
#
#
# class Symmetrization_Net(nn.Module):
#     def __init__(self,d = 10):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Sequential(
#             nn.Linear(d, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4)
#         )
#
#     def forward(self, x):
#         N,_ = x.size()
#         x_ = torch.zeros_like(self.linear(x))
#         elemnts = [k for k in range(N)]
#         for perm in permutations(elemnts):
#             x_ += self.linear(x[perm,:])
#         return x_
#
#
#
# class Sampled_Symmetrization_Net(nn.Module):
#     def __init__(self,d = 10,num_samples = 20):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Sequential(
#             nn.Linear(d, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4)
#         )
#         self.num_samples = num_samples
#
#     def forward(self, x):
#         N,_ = x.size()
#         x_ = torch.zeros_like(self.linear(x))
#         it = create_permutations_sampled(x,self.num_samples)
#         for perm in it:
#             x_ += self.linear(x[perm,:])
#
#         return x_ / self.num_samples
#
#
# class Linear_eq_layer(nn.Module):
#     def __init__(self, d_in=10, d_hidden=32):
#         super().__init__()
#         self.w1 = nn.Linear(d_in,d_hidden)
#         self.w2 = nn.Linear(d_in,d_hidden)
#
#
#     def forward(self, x):  # x is (n, d_in)
#         return self.w1(x) + self.w2(torch.unsqueeze(torch.sum(x,dim=0),dim=0))
#
#
# class Linear_eq_Net(nn.Module):
#     def __init__(self, d_in=10, d_hidden=32 , d_out = 4):
#         super().__init__()
#         self.equiv = nn.Sequential(Linear_eq_layer(d_in,d_hidden),
#                                    nn.ReLU(),
#                                    Linear_eq_layer(d_hidden,d_hidden)
#         )
#
#         self.post_pool = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(d_hidden, d_out))
#
#
#     def forward(self, x):  # x is (n, d_in)
#         x_eq = self.equiv(x)
#         x_pool = x_eq.sum(dim=0)
#         return self.post_pool(x_pool)
#
#
# class AugmentedInvariantNet(nn.Module):
#     def __init__(self, d=10, d_hidden=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             #nn.Flatten(),  # input shape (n, d) → (n*d,)
#             nn.Linear(d , d_hidden),  # assuming n = 10
#             nn.ReLU(),
#             nn.Linear(d_hidden, 1)
#         )
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     def forward(self, x):
#         return self.net(x)