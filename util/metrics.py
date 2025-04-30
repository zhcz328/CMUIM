"""
Metrics utilities for S3UM architecture.

This module implements various metrics for training and evaluation.
"""

import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training and evaluation.
    """

    def __init__(self, name, fmt=':f'):
        """
        Initialize average meter.

        Args:
            name (str): Name of the metric
            fmt (str): Format string for printing
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset all metrics.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update metrics with new value.

        Args:
            val (float): Value to update with
            n (int): Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """
        String representation of average meter.
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def calculate_alp(task_aware_accs, task_agnostic_accs):
    """
    Calculate Average Linear Precision (ALP) metric.

    ALP is the mean of task-aware and task-agnostic accuracies
    across all tasks.

    Args:
        task_aware_accs (list): List of task-aware accuracies
        task_agnostic_accs (list): List of task-agnostic accuracies

    Returns:
        alp (float): Average Linear Precision
    """
    # Ensure lists have same length
    assert len(task_aware_accs) == len(task_agnostic_accs), \
        "Task-aware and task-agnostic accuracy lists must have same length"

    # Calculate mean of both accuracy types
    return np.mean([np.mean(task_aware_accs), np.mean(task_agnostic_accs)])


def calculate_forgetting_rate(accuracy_matrix):
    """
    Calculate forgetting rate from accuracy matrix.

    The forgetting rate measures how much knowledge from previous tasks
    is forgotten after learning new tasks.

    Args:
        accuracy_matrix (np.ndarray): Matrix where accuracy_matrix[i, j]
                                     is the accuracy on task j after learning task i

    Returns:
        forgetting_rate (float): Average forgetting rate
    """
    num_tasks = accuracy_matrix.shape[0]

    # Calculate immediate accuracies after learning each task
    immediate_accs = np.diag(accuracy_matrix)

    # Final accuracies for all tasks
    final_accs = accuracy_matrix[-1, :]

    # Calculate forgetting for each task (excluding the last one)
    forgetting_per_task = immediate_accs[:-1] - final_accs[:-1]

    # Average forgetting
    return np.mean(forgetting_per_task)


def calculate_forward_transfer(accuracy_matrix, baseline_accs):
    """
    Calculate forward transfer from accuracy matrix.

    Forward transfer measures how learning new tasks helps with future tasks.

    Args:
        accuracy_matrix (np.ndarray): Matrix where accuracy_matrix[i, j]
                                     is the accuracy on task j after learning task i
        baseline_accs (np.ndarray): Baseline accuracies for each task when
                                   trained from scratch

    Returns:
        forward_transfer (float): Average forward transfer
    """
    num_tasks = accuracy_matrix.shape[0]

    # Final accuracies for all tasks
    final_accs = accuracy_matrix[-1, :]

    # Calculate forward transfer for each task
    forward_transfer_per_task = final_accs - baseline_accs

    # Average forward transfer
    return np.mean(forward_transfer_per_task)


def calculate_task_confusion_matrix(model, classifier, data_loaders, device, num_classes):
    """
    Calculate confusion matrix across tasks for continual learning.

    Args:
        model (nn.Module): Feature extractor model
        classifier (nn.Module): Linear classifier
        data_loaders (list): List of data loaders for each task
        device (torch.device): Device for evaluation
        num_classes (int): Total number of classes across all tasks

    Returns:
        confusion_matrix (np.ndarray): Confusion matrix
    """
    model.eval()
    classifier.eval()

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for task_id, data_loader in enumerate(data_loaders):
            for samples, targets in data_loader:
                samples = samples.to(device)
                targets = targets.cpu().numpy()

                # Extract features
                features = model.forward_features(samples)

                # Forward pass
                outputs = classifier(features)

                # Get predictions
                _, preds = outputs.max(1)
                preds = preds.cpu().numpy()

                # Update confusion matrix
                for target, pred in zip(targets, preds):
                    confusion_matrix[target, pred] += 1

    # Normalize confusion matrix
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums

    return normalized_confusion_matrix