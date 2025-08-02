import torch
from torch import nn
import numpy as np
from file_manager import DATA_DIR, logger
import os.path as osp
import torch.nn.functional as func

class BasePointCloudNet(nn.Module):

    def __init__(self, n_in=256, d_in=3, d_out=40, creates_own_nn=True, device=None, **kwargs):
        """
        Base class for point cloud networks.
        :param n_in: Number of points in the point cloud
        :param d_in: Dimension of each point in the point cloud
        :param d_out: Dimension of the output (number of classes)
        :param creates_own_nn:  True for inheriting classes that create their own neural network
        :param kwargs: nn.Module initialization parameters (e.g., device, dtype)
        """
        super().__init__(**kwargs)
        self.n = n_in  # Number of points in the point cloud
        self.d_in = d_in  # Dimension of each point in the point cloud
        self.d_input = d_in * n_in  # Input dimension after flattening (n * d)
        self.d_out = d_out  # Number of output (classes)
        self.hidden1 = 64
        self.hidden2 = 128
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing {self.__class__.__name__} with parameters:")
        logger.info(f"n_in: {n_in}, d_in: {d_in}, d_out: {d_out}, device: {str(self.device)}")

        if not creates_own_nn:
            self.flatten = nn.Flatten()
            self.mlp = nn.Sequential(
                nn.Linear(in_features=d_in * n_in, out_features=self.hidden1),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden1, out_features=self.hidden2),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden2, out_features=d_out)
            )

        self.to(self.device)

    def process_input(self, x):
        # Default: no permutation handling
        return x

    def forward(self, x):
        x = self.process_input(x)
        x = self.flatten(x)
        if self.training:
            return self.mlp(x)
        else:
            return func.softmax(self.mlp(x), dim=-1)  # Apply softmax to the output

    def save(self, path=None, version=None):
        """Save model by it's name"""
        if path is None:
            model_name = self.__class__.__name__
            if version is not None:
                model_name += f"_{version}"
            path = osp.join(DATA_DIR, f"{model_name}.pth")
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path=None, version=None):
        """Load model from by it's name"""
        if path is None:
            model_name = self.__class__.__name__
            if version is not None:
                model_name += f"_{version}"
            path = osp.join(DATA_DIR, f"{model_name}.pth")
        if not osp.isfile(path):
            raise RuntimeError(f"Model file does not exist: {path}")
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(self.device)
        logger.info(f"Model loaded from {path}")
        self.eval()  # Set the model to evaluation mode


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
        res = torch.zeros((self.d_out), device=self.device)
        for perm in torch.permutations(torch.arange(self.n)):
            res += self.mlp(self.flatten(x[list(perm), :]))
        return res / np.prod(range(1, self.n+1))

class SampledSymmetrizationNet(BasePointCloudNet):

    def forward(self, x):
        n_samples = 20
        res = torch.zeros((self.d_out), device=self.device)
        for i in range(n_samples):
            perm = torch.randperm(self.n)
            res += self.mlp(self.flatten(x[perm, :]))

        return res / n_samples


class LinearEquivariantLayer(BasePointCloudNet):
    def __init__(self, **kwargs):
        super().__init__(n_in=1, **kwargs)
        self.w1 = nn.Linear(self.d_in, self.d_out, device=self.device)
        self.w2 = nn.Linear(self.d_in, self.d_out, device=self.device)

    def forward(self, x):
        # x: (*, n, 3)
        return self.w1(x) + self.w2(torch.sum(x, dim=-2, keepdim=True))  # output: (n, d_out)

class InvariantNet(BasePointCloudNet):
    def __init__(self, **kwargs):
        super().__init__(n_in = 1, **kwargs)
        self.w = nn.Linear(self.d_in, self.d_out, device=self.device)

    def forward(self, x):
        # x: (*, n, 3)
        return self.w(torch.sum(x, dim=-2))  # output: (n, d_out)

class Permute(nn.Module):
    def __init__(self):
        super().__init__()


class BatchNorm1dWithPermute(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_channels)

    def forward(self, x):
        dims = list(range(x.dim()))
        dims[-2], dims[-1] = dims[-1], dims[-2]
        y = self.bn(x.permute(*dims))
        return y.permute(*dims)


class LinearEquivariantNet(BasePointCloudNet):
    def __init__(self, n_in=256, d_in=3, d_out=40):
        super().__init__(n_in, d_in, d_out)
        hidden1, hidden2 = self.hidden1, self.hidden2
        self.hidden3 = hidden3 = 2 * hidden2
        self.hidden3 = hidden3
        self.equivariant_mpl = nn.Sequential(
            LinearEquivariantLayer(d_in=d_in, d_out=hidden1),
            BatchNorm1dWithPermute(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            LinearEquivariantLayer(d_in=hidden1, d_out=hidden2),
            BatchNorm1dWithPermute(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            # LinearEquivariantLayer(d_in=hidden2, d_out=hidden3),
            # BatchNorm1dWithPermute(hidden3),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.invariant_mpl = nn.Sequential(
            # nn.Linear(in_features=2*hidden3, out_features=hidden3),
            # nn.BatchNorm1d(hidden3),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(in_features=hidden3, out_features=hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden2, out_features=self.d_out)
        )
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def process_input(self, x):
        if x.dim() == 2:
            # If input is 2D, reshape to 3D with n=1
            x = x.unsqueeze(0)  # shape: [1, n, d]

        # # x shape: [*, n, d]
        # if x.shape[-2] > self.n:
        #     # take up to self.n points
        #     x = x[..., :self.n, :]
        return x


    def forward(self, x):
        x = self.process_input(x)
        res_equivariant = self.equivariant_mpl(x)  # res: (n, hidden3)
        # Compute mean and max pooling
        mean_pooled = res_equivariant.mean(dim=-2)
        max_pooled, _ = res_equivariant.max(dim=-2)
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # shape: (batch, 2*hidden3 = hidden4)
        # Adjust invariant_mpl input size if needed
        result = self.invariant_mpl(pooled)  # (batch, d_out)
        if self.training:
            return result
        else:
            return func.softmax(result, dim=-1)



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
#         norms = torch.norm(x, d_in=0)
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
#         return self.w1(x) + self.w2(torch.unsqueeze(torch.sum(x,d_in=0),d_in=0))
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
#         x_pool = x_eq.sum(d_in=0)
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
