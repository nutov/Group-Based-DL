import torch
from torch import nn
import numpy as np
from file_manager import DATA_DIR, logger, TimeIt
import os.path as osp
import torch.nn.functional as func

class BasePointCloudNet(nn.Module):

    def __init__(self, n_in=256, d_in=3, d_out=40, creates_weights=True, device=None,  use_augmentation=None, **kwargs):
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
        self.hidden3 = 256
        self.n_samples = 2 ** 4  # must be a power of 2 and less than n_in
        self.use_augmentation = use_augmentation
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing {self.__class__.__name__} with parameters:")
        logger.info(f"n_in: {n_in}, d_in: {d_in}, d_out: {d_out}, device: {str(self.device)}")
        # Not used for the core MLP path anymore, but kept for other usages
        self.flatten = nn.Flatten(start_dim=-2)

        if creates_weights:            
            self.mlp = nn.Sequential(
                nn.Linear(in_features=d_in * n_in, out_features=self.hidden1),  # (d_in * n_in, hidden1)
                nn.BatchNorm1d(self.hidden1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.hidden1, out_features=self.hidden2),  # (hidden1, hidden2)
                nn.BatchNorm1d(self.hidden2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.hidden2, out_features=self.hidden3),  # (hidden2, hidden3)
                nn.BatchNorm1d(self.hidden3),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(in_features=self.hidden3, out_features=self.hidden2),  # (hidden3, hidden2)
                nn.BatchNorm1d(self.hidden2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.hidden2, out_features=self.d_out),  # (hidden2, d_out)
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
        # Default: input came as (..., n, d) and are fed to the mlp as (..., n * d)
        return x.flatten(start_dim=-2)

    def forward(self, x):
        _x = self.process_input(x)
        y = self.mlp(_x)
        if self.training:
            return y
        else:
            return func.softmax(y, dim=-1)  # Apply softmax to the output

    @TimeIt
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

        return self

    def number_of_parameters(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"{self.__class__.__name__}(n_in={self.n}, d_in={self.d_in}, "
                f"d_out={self.d_out}, device={self.device}, "
                f"number_of_parameters={self.number_of_parameters()})")

    def structure(self):
        """Return a string representation of the model architecture"""
        return str(self)


class CanonizationNet(BasePointCloudNet):
    """Canonicalize input by sorting it according to the norms of the vectors"""
    def process_input(self, x):
        # x shape: (..., n, d)
        # Compute point-wise canonization
        norms = torch.linalg.norm(x, dim=-1)  # (..., n)
        # Sort points by their norms (descending) and gather along the points axis (-2)
        _, indices = torch.sort(norms, dim=-1, descending=True, stable=True)  # (..., n)
        idx_expanded = indices.unsqueeze(-1).expand_as(x)  # (..., n, d)
        x_canon = torch.gather(x, dim=-2, index=idx_expanded)  # shape: (..., n, d)
        # Flatten the input for the MLP
        return self.flatten(x_canon)  # shape: (..., n * d)


class SymmetrizationNet(BasePointCloudNet):

    def permutations_generator(self):
        """
        Generate cyclic permutations of the cyclic group C_n limited to run self.n_samples times.
        """
        identity = torch.arange(self.n, device=self.device)
        for i in range(self.n_samples):
            yield (identity + i) % self.n


    def forward(self, x):
        # x: (..., n, d)
        y = self.flatten(x)  # (..., n * d)
        res = self.mlp(y)  # (..., d_out)
        for i, perm in enumerate(self.permutations_generator()):  # perm shaped (n)
            if i == 0:
                continue
            idx_expanded = perm.view(*([1] * (x.dim() - 2)), -1, 1).expand_as(x)
            x_perm = torch.gather(x, dim=-2, index=idx_expanded)
            res += self.mlp(self.flatten(x_perm))
        return res / self.n_samples

class SampledSymmetrizationNet(SymmetrizationNet):

    def permutations_generator(self):
        """
        Generate samples of permutations of S_n
        """
        for _ in range(self.n_samples):
            yield torch.randperm(self.n, device=self.device)


class LinearEquivariantLayer(BasePointCloudNet):
    def __init__(self, **kwargs):
        super().__init__(n_in=1, creates_weights=False, **kwargs)
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
        hidden1, hidden2, hidden3 = self.hidden1, self.hidden2, self.hidden3
        self.hidden3 = hidden3 = 2 * hidden2
        self.equivariant_mpl = nn.Sequential(
            LinearEquivariantLayer(d_in=d_in, d_out=hidden1),  # (d_in, hidden1)  input: (n, d_in), output: (n, hidden1)
            BatchNorm1dWithPermute(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            LinearEquivariantLayer(d_in=hidden1, d_out=hidden2),  # (hidden1, hidden2)
            BatchNorm1dWithPermute(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            LinearEquivariantLayer(d_in=hidden2, d_out=hidden3),  # (hidden2, hidden3)
            BatchNorm1dWithPermute(hidden3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.invariant_mpl = nn.Sequential(
            nn.Linear(in_features=2*hidden3, out_features=hidden3),  # (2*hidden3, hidden3)
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden3, out_features=hidden2),  # (hidden3, hidden2)
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden2, out_features=self.d_out)  # (hidden2, d_out)
        )
        self._init_weights()
        self.to(self.device)



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
