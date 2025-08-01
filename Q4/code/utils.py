import torch
import torch.nn.functional as func
import numpy as np
from file_manager import logger
import models
import viz
import make_plots
import matplotlib.pyplot as plt
from file_manager import SaveFig
from itertools import permutations

DEBUG_MODE = True

def create_permutations_sampled(x:torch.tensor,K:int):
    N = x.size()[0]
    for _ in range(K):
        yield np.random.permutation(N)


def create_permutations2(n:int):
    """
    Create generator for all permutations of n elements, which runs in a random order
    """
    elements = list(range(n))
    while True:
        perm = np.random.permutation(elements)
        yield perm
#
#
# def test_canonization_net(net, d=10, n=20, tol=1e-5):
#     x = torch.randn(n, d)
#     perm = torch.randperm(n)
#     x_perm = x[perm]
#
#     y = net(x)
#     y_perm = net(x_perm)
#
#     # Invariant: output should not change
#     return torch.allclose(y, y_perm, atol=tol)
#
#
# def test_symmetrization_net(SymmetrizationNet, d=3, n=5, tol=1e-5):  # small n for factorial runtime
#     net = SymmetrizationNet(d=d)
#     x = torch.randn(n, d)
#     perm = torch.randperm(n)
#     x_perm = x[perm,:]
#
#     y = net(x)
#     y_perm = net(x_perm)
#
#     return torch.allclose(y, y_perm, atol=tol)
#
#
# def test_sampled_symmetrization_net(SampledSymmetrizationNet, d=5, n=50, num_samples=50, tol=1e-5):
#     net = SampledSymmetrizationNet(d=d,num_samples = num_samples)
#     x = torch.randn(n, d)
#     perm = torch.randperm(n)
#     x_perm = x[perm]
#
#     y = net(x)
#     y_perm = net(x_perm)
#
#     return torch.allclose(y, y_perm, atol=tol)
#
#
def test_equivariance_equivariant_layer(net: models.BasePointCloudNet, input:torch.tensor, tol=1e-5):

    x = input
    n = x.shape[0]
    perm = torch.randperm(n)

    x_perm = x[perm, :]

    y = net(x)
    y_perm = net(x_perm)
    logger.info(f"Permuted output distance: {torch.norm(y - y_perm)}")

    return torch.allclose(y[perm], y_perm, atol=tol)
#
#
# def test_equivariance(Linear_eq_layer,d_in=10, n=10, tol=1e-5):
#     net = Linear_eq_layer(d_in=d_in)
#     x = torch.randn(n, d_in)
#     perm = torch.randperm(n)
#     x_perm = x[perm]
#
#     y = net(x)[perm]
#     y_perm = net(x_perm)
#
#     return torch.allclose(y, y_perm, atol=tol)
#
# def test_invariant_net(model_class, d_in=10, n=6, tol=1e-5):
#     net = model_class(d_in=d_in)
#     x = torch.randn(n, d_in)
#     x_perm = x[torch.randperm(n)]
#     y = net(x)
#     y_perm = net(x_perm)
#     return torch.allclose(y, y_perm, atol=tol)
#
#
# def run_test(test_args:tuple,num_tests = 100):
#     res = 0
#     test_func,net = test_args
#     for _ in range(num_tests):
#         if not test_func(net):
#             res+=1
#     return res/num_tests
#
#
# def compute_variance_target(x):
#     return torch.unsqueeze(x.var(dim=1, unbiased=True),dim=1)
#
# def custom_loss(outputs, targets):
#     prec = 1 / torch.var(outputs)
#     not_mse = 0.5 * prec * torch.mean(torch.pow(outputs - targets, 2))
#     return not_mse - 0.5 * torch.log(prec)
#
#
# def _basic_training(model, optimizer, target_label, in_data):
#     # check that the model updates
#     optimizer.zero_grad()
#     output = model(in_data)
#     loss = func.cross_entropy(output, target_label)
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
# def _augmentation_f(data):
#     # raise NotImplemented
#     return data  #TODO


def train(model: models.BasePointCloudNet, optimizer, train_loader, test_loader=None, epochs=100, augmentations=0, verbose=False, use_augmentation=False):
    """
    Train a model with optional data augmentation.
    :param model:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param augmentations:
    :param verbose:
    :param use_augmentation:
    :return:
    model, train_losses, test_losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    test_losses = []
    model.execution_phase = False


    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}")
        model.train()
        running_loss = 0.0
        num_batches = 0
        for inputs, target_labels in train_loader:
            inputs, target_labels = inputs.to(device), target_labels.to(device)
            if use_augmentation:
                for _ in range(augmentations):
                    aug_inputs = _augmentation_f(inputs)
                    loss = _basic_training(model, optimizer, target_labels, aug_inputs)
                    running_loss += loss
                    num_batches += 1
            else:
                loss = _basic_training(model, optimizer, target_labels, inputs)

                running_loss += loss
                num_batches += 1
        if num_batches == 0:
            logger.info("Warning: No batches processed in this epoch. Check your data loader.")
            continue

        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        avg_test_loss = None
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for inputs, target_labels in test_loader:
                    inputs, target_labels = inputs.to(device), target_labels.to(device)
                    outputs = model(inputs)
                    loss = func.cross_entropy(outputs, target_labels)
                    test_loss += loss.item()
                    test_batches += 1
            avg_test_loss = test_loss / max(1, test_batches)
            test_losses.append(avg_test_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.8f}")
        if avg_test_loss is not None:
            logger.info(f" | Test Loss: {avg_test_loss:.8f}")

        model.execution_phase = True
    return model, train_losses, test_losses

# def test_variance_invariance(model,d=50, tol=1e-1, num_tests=100):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     x = torch.randn((100,d)).to(device)
#     model.eval()
#     with torch.no_grad():
#         y_ref = model(x)
#         for _ in range(num_tests):
#             perm = torch.randperm(x.size(0))
#             y_alt = model(x[perm])
#             if not torch.allclose(y_ref[perm], y_alt, atol=tol):
#                 return False
#     return True

