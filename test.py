import matplotlib.pyplot as plt
from flex.data import FedDataDistribution, FedDatasetConfig, Dataset
from torchvision import datasets, transforms
import tarfile
import numpy as np
import torch
import os
import pickle
from torch import nn as nn, optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

cifar_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

training_data = datasets.CIFAR10(
    root=".", train=True, download=False, transform=cifar_transforms
)

test_data = datasets.CIFAR10(
    root=".", train=True, download=False, transform=None
)

config = FedDatasetConfig(seed=0)
config.replacement = False
config.n_nodes = 100

flex_dataset = FedDataDistribution.from_config(
    centralized_data=Dataset.from_torchvision_dataset(training_data), config= config
)

figure = plt.figure(figsize=(8,8))
cols, rows = 32, 32
for i in range(1, cols * rows + 1):
    sample = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
plt.show()
#Federating process











class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 2

"""
fcd_torch = Dataset.from_torchvision_dataset(dataset)

config_torch = FedDatasetConfig(seed=0, n_nodes=2, replacement=False)

"""
