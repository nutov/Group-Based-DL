import matplotlib
import os
if os.name == 'nt':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')

import models

import numpy as np
import unittest
import data
import torch

class TestModelsPermutationInvariance(unittest.TestCase):
    """
    Test the permutation invariance of the models.
    """

    # def test_permutation_invariance_of_linear_net(self):
    #     equivariant_model = models.LinearEquivariantNet()
    #     equivariant_model.load()
    #
    #     equivariant_model.eval()
    #     self.train_dataset = data.TrainDataset()
    #
    #     input_data, true_label = self.train_dataset[0]
    #     output_data = equivariant_model(input_data)
    #
    #     n = input_data.shape[0]
    #     for _ in range(10):
    #         perm = np.random.Generator.permutation(range(n))
    #         permuted_input = input_data[perm, :]
    #         permuted_output = equivariant_model(permuted_input)
    #         print(f"Permuted output distance: {torch.norm(output_data - permuted_output)}")
    #     self.assertTrue(torch.allclose(output_data, permuted_output, atol=1e-5),
    #                     "Output data is not invariant to permutations.")

    def test_equivariant_layer(self):
        n = 256
        d = 3
        tol = 1e-5
        net = models.LinearEquivariantLayer(d_in=d, d_out=40)
        # make the weight matrix to be random
        net.w1.weight.data = torch.randn_like(net.w1.weight.data, dtype=torch.float64)
        net.w2.weight.data = torch.randn_like(net.w2.weight.data, dtype=torch.float64)
        net.w1.bias.data = torch.randn_like(net.w1.bias.data, dtype=torch.float64)
        net.w2.bias.data = torch.randn_like(net.w2.bias.data, dtype=torch.float64)
        net.eval()
        for i in range(10):
            x = torch.randn((n, d), dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[perm]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y[perm], y_perm, atol=tol),
                            f"Equivariant layer failed for iteration {i}: {torch.norm(y[perm] - y_perm)}")

        n_batch = 32
        for i in range(10):
            x = torch.randn(n_batch, n, d, dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[:, perm, :]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y[:, perm, :], y_perm, atol=tol),
                            f"Equivariant layer failed for iteration {i}: {torch.norm(y[:,perm,:] - y_perm)}")

        return True

    def test_invariant_net(self):
        n = 256
        d = 128
        tol = 1e-5
        net = models.LinearInvariantNet(d_in=d, d_out=40)
        # make the weight matrix to be random
        net.w.weight.data = torch.randn_like(net.w.weight.data, dtype=torch.float64)
        net.w.bias.data = torch.randn_like(net.w.bias.data, dtype=torch.float64)
        net.eval()

        for i in range(10):
            x = torch.randn(n, d, dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[perm]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y, y_perm, atol=tol),
                            f"Invariant layer failed for iteration {i}: {torch.norm(y - y_perm)}")

        n_batch = 32
        for i in range(10):
            x = torch.randn((n_batch, n, d), dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[:,perm,:]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y, y_perm, atol=tol),
                            f"Invariant layer failed for iteration {i}: {torch.norm(y - y_perm)}")

        return True

    def test_equivariant_net(self):
        n = 256
        d = 3
        tol = 1e-5
        net = models.LinearEquivariantNet(n_in=n, d_in=d, d_out=40)
        net.eval()
        # make all the parameters to be random
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = torch.randn_like(m.weight.data, dtype=torch.float64)
                if m.bias is not None:
                    m.bias.data = torch.randn_like(m.bias.data, dtype=torch.float64)

        for i in range(10):
            x = torch.randn((n, d), dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[perm]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y, y_perm, atol=tol),
                            f"Equivariant net failed for iteration {i}: {torch.norm(y - y_perm)}")

        n_batch = 32
        for i in range(10):
            x = torch.randn((n_batch, n, d), dtype=torch.float64)
            perm = torch.randperm(n)
            x_perm = x[:, perm, :]

            y = net(x)
            y_perm = net(x_perm)
            self.assertTrue(torch.allclose(y, y_perm, atol=tol),
                            f"Equivariant net failed for iteration {i}: {torch.norm(y - y_perm)}")

        return True
