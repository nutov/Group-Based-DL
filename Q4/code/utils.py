import torch
import torch.nn.functional as F
import numpy as np
from file_manager import logger, SaveData, TimeIt
import models
import viz
import make_plots
import matplotlib.pyplot as plt
from file_manager import SaveFig
from itertools import permutations
import time

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

@TimeIt
def calculate_accuracy_and_loss(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
    """
    Calculate the accuracy and average loss of the model on the given data loader.
    :param model: The model to evaluate.
    :param data_loader: DataLoader containing the dataset.
    :return: Tuple (accuracy, avg_loss).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0.0
    total = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, target_labels in data_loader:
            inputs, target_labels = inputs.to(device), target_labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, target_labels)
            _, predicted = torch.max(outputs, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            total_loss += loss.item() * target_labels.size(0)

    accuracy = correct / total if total > 0. else 0.0
    avg_loss = total_loss / total if total > 0. else float('inf')
    logger.info(f"Accuracy, Loss of {model.__class__.__name__} on {data_loader.dataset.__class__.__name__} | {accuracy:.4f}, Avg Loss: {avg_loss:.4f}")

    return accuracy, avg_loss

#
#
def test_equivariance_equivariant_layer(net: models.BasePointCloudNet, input:torch.tensor, tol=1e-3):

    x = input  # shape [n, d]
    n = x.shape[0]
    perm = torch.randperm(n)
    x_perm = x[perm]

    y = net(x)  # shape [d_out]
    y_perm = net(x_perm)  # shape [d_out]
    logger.info(f"Permuted output distance: {torch.norm(y - y_perm)}")

    return torch.allclose(y, y_perm, atol=tol)
#
#
def test_equivariant_layer():
    n = 256
    d = 3
    tol = 1e-5
    net = models.LinearEquivariantLayer(d_in=d, d_out=40)
    x = torch.randn(n, d)
    perm = torch.randperm(n)
    x_perm = x[perm]

    y = net(x)[perm]
    y_perm = net(x_perm)

    return torch.allclose(y, y_perm, atol=tol)
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
#     return torch.unsqueeze(x.var(d_in=1, unbiased=True),d_in=1)
#
# def custom_loss(outputs, targets):
#     prec = 1 / torch.var(outputs)
#     not_mse = 0.5 * prec * torch.mean(torch.pow(outputs - targets, 2))
#     return not_mse - 0.5 * torch.log(prec)
#
#
@TimeIt
def _basic_training(model, optimizer, target_label, in_data):
    # check that the model updates
    optimizer.zero_grad()
    output = model(in_data)
    loss = F.cross_entropy(output, target_label)
    loss.backward()
    optimizer.step()
    return loss.item()

def _augmentation_f(data):
    # raise NotImplemented
    return data  #TODO

@TimeIt
def train(model_list,
          optimizer_list,
          train_loader,
          test_loader=None,
          epochs=100,
          augmentations=0,
          use_augmentation=False):
    """
    Train a model with optional data augmentation.
    :param model_list:
    :param optimizer_list:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param augmentations:
    :param use_augmentation:
    :return:
    Trained model, training accuracy, training loss, test accuracy, test loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(model_list)

    train_loss = [[] for _ in range(n_models)]
    test_loss = [[] for _ in range(n_models)]
    train_acc = [[] for _ in range(n_models)]
    test_acc = [[] for _ in range(n_models)]

    # Timing containers
    train_time_per_model_per_epoch = [[] for _ in range(n_models)]           # seconds per epoch per model
    eval_test_time = [[] for _ in range(n_models)]       # seconds to eval on test_loader

    for epoch in range(epochs):

        logger.info(f"Epoch {epoch}")
        for model in model_list:
            model.train()
            model.to(device)

        # Accumulate per-model training time over all batches in this epoch
        per_model_train_times = [0.0 for _ in range(n_models)]

        for inputs, target_labels in train_loader:
            inputs, target_labels = inputs.to(device), target_labels.to(device)
            for i, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
                t0 = time.perf_counter()
                if model.use_augmentation:                    
                    for _ in range(augmentations):
                        aug_inputs = _augmentation_f(inputs)
                        _basic_training(model, optimizer, target_labels, aug_inputs)
                    per_model_train_times[i] += (time.perf_counter() - t0)
                else:
                    _basic_training(model, optimizer, target_labels, inputs)
                    per_model_train_times[i] += (time.perf_counter() - t0)

        # Record training time per model for this epoch
        for i in range(n_models):
            train_time_per_model_per_epoch[i].append(per_model_train_times[i])

        for i, model in enumerate(model_list):
            # calculate metrics and measure eval time (train loader)
            _acc, _loss = calculate_accuracy_and_loss(model, train_loader)
            train_acc[i].append(_acc)
            train_loss[i].append(_loss)
            if test_loader:
                t0_eval_test = time.perf_counter()
                _test_acc, _test_loss = calculate_accuracy_and_loss(model, test_loader)
                eval_test_elapsed = time.perf_counter() - t0_eval_test
                test_acc[i].append(_test_acc)
                test_loss[i].append(_test_loss)
                eval_test_time[i].append(eval_test_elapsed)

            # Log timings summary for this epoch/model
            logger.info(
                f"Timing | Epoch {epoch} | {model.__class__.__name__}: "
                f"train_time={train_time_per_model_per_epoch[i][-1]:.3f}s, "
                f"eval_train_time={eval_train_time[i][-1]:.3f}s" +
                (f", eval_test_time={eval_test_time[i][-1]:.3f}s" if test_loader else "")
            )

    # save timing data
    SaveData(train_time_per_model_per_epoch, "train_time_per_epoch")
    if test_loader:
        SaveData(eval_test_time, "eval_test_time_per_epoch")

    return train_acc, train_loss, test_acc, test_loss

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
