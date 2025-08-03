import re


from collections import defaultdict
import matplotlib.pyplot as plt

def extract_networks_evaluations(log_path):
    """
    Extracts train/test loss and accuracy per epoch per network from the logger file.
    Returns a dictionary: {network_name: {"train_loss": [...], "train_accuracy": [...], "test_loss": [...], "test_accuracy": [...]}}
    """
    networks_evaluations = defaultdict(lambda: {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    })

    train_pattern = re.compile(
        r"Accuracy, Loss of ([A-Za-z0-9_]+) on TrainDataset \| ([0-9.]+), Avg Loss: ([0-9.]+)"
    )
    test_pattern = re.compile(
        r"Accuracy, Loss of ([A-Za-z]+) on TestDataset \| ([0-9.]+), Avg Loss: ([0-9.]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                net_name = train_match.group(1)
                acc = float(train_match.group(2))
                loss = float(train_match.group(3))
                networks_evaluations[net_name]["train_accuracy"].append(acc)
                networks_evaluations[net_name]["train_loss"].append(loss)
            test_match = test_pattern.search(line)
            if test_match:
                net_name = test_match.group(1)
                acc = float(test_match.group(2))
                loss = float(test_match.group(3))
                networks_evaluations[net_name]["test_accuracy"].append(acc)
                networks_evaluations[net_name]["test_loss"].append(loss)

    return dict(networks_evaluations)

# Example usage:
# networks_evaluations = extract_networks_evaluations("path/to/logger.log")


if __name__ == "__main__":

    log_path = "/home/tuvy/Documents/study/deep_and_groups_hw4/Group-Based-DL/Q4/code/results/20250803_2143/logger.log"
    networks_evaluations = extract_networks_evaluations(log_path)
    print(networks_evaluations)


    plt.figure(1)
    plt.plot(networks_evaluations["LinearEquivariantNet"]["train_accuracy"], label='Train Accuracy')
    plt.plot(networks_evaluations["LinearEquivariantNet"]["test_accuracy"], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch for LinearEquivariantNet')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.semilogy(networks_evaluations["LinearEquivariantNet"]["train_loss"], label='Train Loss')
    plt.semilogy(networks_evaluations["LinearEquivariantNet"]["test_loss"], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch for LinearEquivariantNet')
    plt.legend()
    plt.show()

