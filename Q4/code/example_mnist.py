import torch
import torch.nn as nn  # Import neural network module
import torch.optim as optim  # Import optimization module
from torchvision import datasets, transforms  # Import for datasets and transformations

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean 0.5 and standard deviation 0.5
])

# Load MNIST datasets, applying the defined transformations
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create DataLoaders for efficient training and testing data handling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model definition
model = nn.Sequential(
    nn.Flatten(),     # Flatten images into a single vector
    nn.Linear(784, 128),  # Fully connected layer with 128 neurons
    nn.ReLU(),           # ReLU activation for non-linearity
    nn.Linear(128, 10)   # Output layer with 10 neurons (for 10 classes in MNIST)
)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # Common loss function for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training and Evaluation loop
for epoch in range(5):  # Loop for 5 epochs
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):  # Iterate over batches of data
        optimizer.zero_grad()  # Clear gradients from the previous iteration
        output = model(data)  # Forward pass through the model
        loss = loss_fn(output, target)  # Calculate the loss
        loss.backward()  # Compute gradients (backpropagation)
        optimizer.step()  # Update model parameters

    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data, target in test_loader:  # Iterate over test data
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Accumulate test loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()  # Update correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate average test loss
    print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))