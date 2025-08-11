import time
import numpy as np
import torch
import torch.nn.functional as F
import models
from file_manager import logger, SaveData, TimeIt


def create_permutations_sampled(x:torch.Tensor, K:int):
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()

    correct = 0.0
    total = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, target_labels in data_loader:
            # inputs, target_labels = inputs.to(device), target_labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, target_labels)
            _, predicted = torch.max(outputs, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            total_loss += loss.item() * target_labels.size(0)

    accuracy = correct / total if total > 0. else 0.0
    avg_loss = total_loss / total if total > 0. else float('inf')
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
def _basic_training(model :models.BasePointCloudNet, optimizer: torch.optim.Optimizer, target_label, in_data):
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
          test_loader,
          validation_loader,
          epochs,
          k):
    """
    Train a model with optional data augmentation.
    :param model_list:
    :param optimizer_list:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :return:
    Trained model, training accuracy, training loss, test accuracy, test loss, training time, evaluation time.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(model_list)
    stop_training = [False for _ in range(n_models)]  # stop training if True
    train_loss = [[] for _ in range(n_models)]
    test_loss = [[] for _ in range(n_models)]
    validation_loss = [[] for _ in range(n_models)]
    train_acc = [[] for _ in range(n_models)]
    test_acc = [[] for _ in range(n_models)]
    validation_acc = [[] for _ in range(n_models)]

    # Timing containers
    train_time_per_model_per_epoch = [[] for _ in range(n_models)]           # time [sec] per model (first index) per epoch (second index)
    eval_time_per_model_per_epoch = [[] for _ in range(n_models)]            # total eval time (train+val+test) per epoch per model

    for epoch in range(epochs):

        logger.info(f"Epoch {epoch}")
        for model in model_list:
            model.train()
            model.to(device)

        # Accumulate per-model training time over all batches in this epoch
        epoch_train_time_per_model = [0.0 for _ in range(n_models)]

        for inputs, target_labels in train_loader:
            inputs, target_labels = inputs.to(device), target_labels.to(device)
            for i, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
                if stop_training[i]:
                    continue
                t0 = time.perf_counter()
                if model.use_augmentation:                    
                    for _ in range(model.n_samples):
                        aug_inputs = _augmentation_f(inputs)
                        _basic_training(model, optimizer, target_labels, aug_inputs)
                    epoch_train_time_per_model[i] += (time.perf_counter() - t0)
                else:
                    _basic_training(model, optimizer, target_labels, inputs)
                    epoch_train_time_per_model[i] += (time.perf_counter() - t0)

        for i in range(n_models):
            train_time_per_model_per_epoch[i].append(epoch_train_time_per_model[i])

        for i, model in enumerate(model_list):
            # total eval time across train/val/test
            t_eval0 = time.perf_counter()
            # calculate metrics (train)
            msg = f"{model.__class__.__name__} | Epoch {epoch} |  "
            _acc, _loss = calculate_accuracy_and_loss(model, train_loader)
            msg += f"Train accuracy: {_acc:.4f} |  "
            # If the accuracy on the training is almost 1, the loss is very small and in any what not useful, we stop training this model
            # if _acc > 0.99:
            #     stop_training[i] = True
            train_acc[i].append(_acc)
            train_loss[i].append(_loss)
            # calculate metrics (validation)
            _acc, _loss = calculate_accuracy_and_loss(model, validation_loader)
            msg += f"Validation accuracy: {_acc:.4f} |  "
            msg += f"Validation accuracy: {_acc:.4f} |  "
            if (k==-1) and (_acc > 0.8) or (k != -1 and _acc > 0.6):
                stop_training[i] = True
            validation_acc[i].append(_acc)
            validation_loss[i].append(_loss)
            # calculate metrics (test)
            _acc, _loss = calculate_accuracy_and_loss(model, test_loader)
            msg += f"Test accuracy: {_acc:.4f} |  "
            test_acc[i].append(_acc)
            test_loss[i].append(_loss)
            logger.info(msg=msg)
            # close eval timer
            eval_time_per_model_per_epoch[i].append(time.perf_counter() - t_eval0)

    # write to loger finale results per model, including: average  training time, final accuracy on: training set, validation set, and test set,
    average_training_time = np.mean(train_time_per_model_per_epoch, axis=1)  # train_time_per_model_per_epoch is n_models x n_epochs
    average_eval_time = np.mean(eval_time_per_model_per_epoch, axis=1)       # average total eval time per model
    logger.info(f"Final results for k = {k}:\n")
    for i, model in enumerate(model_list):
        logger.info(f"Model: {model.__class__.__name__} |\t\t"
                    f"average training time: {average_training_time[i]:.8f} [sec] |\t\t"
                    f"average evaluation time: {average_eval_time[i]:.8f} [sec] |\t\t"
                    f"number of epochs: {len(train_acc[i])} |\t\t"
                    f"training accuracy: {train_acc[i][-1]:.4f} |\t\t"
                    f"validation accuracy: {validation_acc[i][-1]:.4f} |\t\t"
                    f"test accuracy: {test_acc[i][-1]:.4f} |\t\t"
                    f"training loss: {train_loss[i][-1]:.4f} |\t\t"
                    f"validation loss: {validation_loss[i][-1]:.4f} |\t\t"
                    f"test loss: {test_loss[i][-1]:.4f} |\t\t"
                    f"out of time: {not stop_training[i]}")
    # save data
    k_tag = 'all' if k == -1 else str(k)
    SaveData(train_time_per_model_per_epoch, f"train_time_per_epoch_k_{k_tag}")
    SaveData(eval_time_per_model_per_epoch, f"eval_time_per_epoch_k_{k_tag}")
    SaveData(train_acc, f"train_accuracy_per_epoch_k_{k_tag}")
    SaveData(train_loss, f"train_loss_per_epoch_k_{k_tag}")
    SaveData(validation_acc, f"validation_accuracy_per_epoch_k_{k_tag}")
    SaveData(validation_loss, f"validation_loss_per_epoch_k_{k_tag}")
    SaveData(test_acc, f"test_accuracy_per_epoch_k_{k_tag}")
    SaveData(test_loss, f"test_loss_per_epoch_k_{k_tag}")
    # save all models
    for model in model_list:
        model.save(version=k_tag)

    return train_acc, train_loss, test_acc, test_loss


def test_variance_invariance(model: models.BasePointCloudNet, 
                             d=50, tol=1e-1, num_tests=100):
    x = torch.randn(size=(model.n, model.d_in), device=model.device, dtype=model.dtype)
    model.eval()
    with torch.no_grad():
        y_ref = model(x)
        for _ in range(num_tests):
            perm = torch.randperm(x.size(0))
            y_alt = model(x[perm])
            if not torch.allclose(y_ref[perm], y_alt, atol=tol):
                return False
    return True
