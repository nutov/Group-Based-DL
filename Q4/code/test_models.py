import models

import numpy as np
import unittest
import data
import torch

class TestModelsPermutationInvariance(unittest.TestCase):
    """
    Test the permutation invariance of the models.
    """

    def test_permutation_invariance_of_linear_net(self):
        equivariant_model = models.LinearEquivariantNet()
        equivariant_model.load()

        equivariant_model.eval()
        self.train_dataset = data.TrainDataset()

        input_data, true_label = self.train_dataset[0]
        output_data = equivariant_model(input_data)

        n = input_data.shape[0]
        for _ in range(10):
            perm = np.random.Generator.permutation(range(n))
            permuted_input = input_data[perm, :]
            permuted_output = equivariant_model(permuted_input)
            print(f"Permuted output distance: {torch.norm(output_data - permuted_output)}")
        self.assertTrue(torch.allclose(output_data, permuted_output, atol=1e-5),
                        "Output data is not invariant to permutations.")
