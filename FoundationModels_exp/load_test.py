#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:21:43 2025

@author: MYQUEEN
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set up resizing for foundation models (e.g., CLIP, ViT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Compatible with ViT/CLIP
    transforms.ToTensor()
])

# Load CIFAR-10
cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Load STL-10
stl_train = datasets.STL10(root="./data", split='train', download=True, transform=transform)
stl_test = datasets.STL10(root="./data", split='test', download=True, transform=transform)

# Create DataLoaders (for feature extraction later)
cifar_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=False)
cifar_test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)
stl_train_loader = DataLoader(stl_train, batch_size=64, shuffle=False)
stl_test_loader = DataLoader(stl_test, batch_size=64, shuffle=False)

# Plot example images
def show_samples(dataset, title):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        image, label = dataset[i]
        axs[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axs[i].set_title(f"Label: {label}")
        axs[i].axis("off")
    plt.suptitle(title)
    plt.show()

show_samples(cifar_train, "CIFAR-10 Samples")
show_samples(stl_train, "STL-10 Samples")
