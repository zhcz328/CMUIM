"""
Data loading utilities for S3UM architecture.

This module implements dataset loaders for ultrasound images in
both standard and continual learning settings.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
from collections import defaultdict


class CustomImageFolderDataset(Dataset):
    """
    Custom dataset class for handling image folders with additional features.

    Enhances the standard ImageFolder dataset with:
    - Tracking class counts
    - Handling corrupted images
    - More flexible data transformations
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Transform to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform

        if root_dir and os.path.exists(root_dir):
            self.classes = sorted(entry.name for entry in os.scandir(root_dir)
                                  if entry.is_dir())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.images = self._load_images()
            self.class_counts = self._count_images_per_class()
        else:
            self.classes = []
            self.class_to_idx = {}
            self.images = []
            self.class_counts = {}

    def _load_images(self):
        """
        Load image paths and labels.

        Returns:
            images (list): List of (image_path, label) tuples
        """
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    images.append((img_path, self.class_to_idx[class_name]))
        return images

    def _count_images_per_class(self):
        """
        Count the number of images per class.

        Returns:
            class_counts (dict): Dictionary mapping class names to image counts
        """
        class_counts = {cls_name: 0 for cls_name in self.classes}
        for _, label in self.images:
            class_counts[self.classes[label]] += 1
        return class_counts

    def __len__(self):
        """Return the total number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an item by index.

        Args:
            idx (int): Index of the item

        Returns:
            image (torch.Tensor): Transformed image
            label (int): Class label
        """
        img_path, label = self.images[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Remove corrupted image from dataset
            self.images.pop(idx)
            # Return a placeholder or a different image
            if len(self.images) > 0:
                new_idx = idx % len(self.images)
                return self.__getitem__(new_idx)
            else:
                # Create a blank image as fallback
                blank = torch.zeros(3, 224, 224)
                return blank, 0


class ConcatCustomImageFolderDataset(Dataset):
    """
    Concatenate multiple CustomImageFolderDataset instances.

    This is useful for combining datasets from different tasks
    or domains in continual learning settings.
    """

    def __init__(self, datasets):
        """
        Initialize the concatenated dataset.

        Args:
            datasets (list): List of Dataset objects to concatenate
        """
        self.datasets = datasets

        if not datasets:
            self.total_length = 0
            return

        # Filter out empty datasets
        self.datasets = [ds for ds in datasets if isinstance(ds, Dataset) and len(ds) > 0]

        # Calculate total length
        self.total_length = sum(len(ds) for ds in self.datasets)

        # Calculate dataset endpoints for indexing
        self.endpoints = [0]
        for ds in self.datasets:
            self.endpoints.append(self.endpoints[-1] + len(ds))

    def __len__(self):
        """Return the total number of images."""
        return self.total_length

    def __getitem__(self, idx):
        """
        Get an item by index from the appropriate dataset.

        Args:
            idx (int): Index of the item

        Returns:
            image (torch.Tensor): Transformed image
            label (int): Class label
        """
        if self.total_length == 0:
            raise IndexError("Empty dataset")

        # Find the dataset that contains this index
        dataset_idx = 0
        while dataset_idx < len(self.endpoints) - 1 and idx >= self.endpoints[dataset_idx + 1]:
            dataset_idx += 1

        # Get the item from the appropriate dataset
        relative_idx = idx - self.endpoints[dataset_idx]
        return self.datasets[dataset_idx][relative_idx]


def create_dataset_loaders(root_dir, batch_size=64, transform=None,
                           num_workers=4, val_split=0.2, seed=42):
    """
    Create DataLoader objects for training and validation.

    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        transform (callable, optional): Transform to apply to images
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create dataset
    dataset = CustomImageFolderDataset(root_dir, transform=transform)

    # Split into train and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    val_size = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def create_continual_datasets(data_path, n_tasks, batch_size=64, buffer_size=500,
                              num_workers=4, seed=42):
    """
    Create datasets for continual learning across multiple tasks.

    Args:
        data_path (str): Base directory for task datasets
        n_tasks (int): Number of tasks
        batch_size (int): Batch size
        buffer_size (int): Number of samples to keep in memory buffer
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility

    Returns:
        train_loaders (list): List of training DataLoader objects
        val_loaders (list): List of validation DataLoader objects
        buffer_loaders (list): List of buffer DataLoader objects
        test_loaders (list): List of test DataLoader objects
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Assume data is organized in task directories
    task_dirs = [os.path.join(data_path, f"task{i}") for i in range(n_tasks)]

    # Verify task directories exist
    valid_task_dirs = []
    for task_dir in task_dirs:
        if os.path.exists(os.path.join(task_dir, "train")) and os.path.exists(os.path.join(task_dir, "val")):
            valid_task_dirs.append(task_dir)
        else:
            print(f"Warning: Task directory {task_dir} is missing train or val subdirectories")

    if not valid_task_dirs:
        raise ValueError(f"No valid task directories found in {data_path}")

    # Create datasets and loaders for each task
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for task_dir in valid_task_dirs:
        train_dir = os.path.join(task_dir, "train")
        val_dir = os.path.join(task_dir, "val")
        test_dir = os.path.join(task_dir, "test") if os.path.exists(os.path.join(task_dir, "test")) else val_dir

        train_datasets.append(CustomImageFolderDataset(train_dir, transform=train_transform))
        val_datasets.append(CustomImageFolderDataset(val_dir, transform=val_transform))
        test_datasets.append(CustomImageFolderDataset(test_dir, transform=val_transform))

    # Create buffer datasets
    buffer_datasets = [_create_buffer_dataset(train_datasets, i, buffer_size, train_transform)
                       for i in range(len(valid_task_dirs))]

    # Create data loaders
    train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
                     for ds in train_datasets]

    val_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
                   for ds in val_datasets]

    test_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
                    for ds in test_datasets]

    buffer_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True)
                      for ds in buffer_datasets]

    return train_loaders, val_loaders, buffer_loaders, test_loaders


def _create_buffer_dataset(train_datasets, current_task_idx, buffer_size, transform):
    """
    Create a buffer dataset for continual learning.

    Args:
        train_datasets (list): List of training datasets
        current_task_idx (int): Index of the current task
        buffer_size (int): Number of samples to keep per task
        transform (callable): Transform to apply to images

    Returns:
        buffer_dataset (Dataset): Buffer dataset for continual learning
    """
    if current_task_idx == 0:
        # No buffer for the first task
        return None

    buffer_datasets = []

    # For each previous task
    for task_idx in range(current_task_idx):
        task_dataset = train_datasets[task_idx]

        # Group samples by class
        class_to_samples = defaultdict(list)
        for idx in range(len(task_dataset)):
            _, label = task_dataset[idx]
            class_to_samples[label].append(idx)

        # Select balanced samples from each class
        selected_indices = []
        samples_per_class = buffer_size // len(class_to_samples)

        for label, indices in class_to_samples.items():
            random.shuffle(indices)
            selected_indices.extend(indices[:samples_per_class])

        # Add remaining slots if needed
        remaining = buffer_size - len(selected_indices)
        if remaining > 0:
            all_indices = list(range(len(task_dataset)))
            random.shuffle(all_indices)

            # Filter out already selected indices
            all_indices = [idx for idx in all_indices if idx not in selected_indices]

            # Add remaining indices
            selected_indices.extend(all_indices[:remaining])

        # Create subset dataset
        buffer_datasets.append(Subset(task_dataset, selected_indices))

    # Combine all buffer datasets
    return ConcatDataset(buffer_datasets)