"""
Test and evaluation utilities for S3UM architecture.

This module implements evaluation procedures for the S3UM framework,
including metrics for continual learning scenarios.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..utils.metrics import AverageMeter


def evaluate(model, data_loader, device):
    """
    Evaluate model reconstruction performance.

    Args:
        model (nn.Module): CMUIM model
        data_loader (DataLoader): Evaluation data loader
        device (torch.device): Device for evaluation

    Returns:
        metrics (dict): Evaluation metrics
    """
    model.eval()

    # Initialize metrics
    recon_losses = AverageMeter("Reconstruction Loss")

    with torch.no_grad():
        for samples, _ in tqdm(data_loader, desc="Evaluating"):
            samples = samples.to(device, non_blocking=True)

            # Forward pass
            loss, _ = model(samples, mask_ratio=0.75)

            # Update metrics
            recon_losses.update(loss.item(), samples.size(0))

    print(f"Evaluation Results:")
    print(f"  Reconstruction Loss: {recon_losses.avg:.4f}")

    metrics = {
        'recon_loss': recon_losses.avg
    }

    return metrics


def evaluate_linear(model, classifier, data_loader, device, task_id=None):
    """
    Evaluate linear classification performance.

    Args:
        model (nn.Module): Feature extractor
        classifier (nn.Module): Linear classifier
        data_loader (DataLoader): Evaluation data loader
        device (torch.device): Device for evaluation
        task_id (int, optional): Task ID for task-aware evaluation

    Returns:
        metrics (dict): Evaluation metrics
    """
    model.eval()
    classifier.eval()

    # Initialize metrics
    losses = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for samples, targets in tqdm(data_loader, desc="Evaluating Linear"):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Extract features
            features = model.forward_features(samples)

            # Forward pass through classifier
            if task_id is not None and hasattr(classifier, 'task_forward'):
                # Task-aware evaluation
                outputs = classifier.task_forward(features, task_id)
            else:
                # Task-agnostic evaluation
                outputs = classifier(features)

            # Calculate loss and accuracy
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # Update metrics
            losses.update(loss.item(), samples.size(0))
            top1.update(acc1.item(), samples.size(0))
            top5.update(acc5.item(), samples.size(0))

    print(f"Linear Evaluation Results:")
    print(f"  Loss: {losses.avg:.4f}")
    print(f"  Acc@1: {top1.avg:.4f}")
    print(f"  Acc@5: {top5.avg:.4f}")

    metrics = {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg
    }

    return metrics


def evaluate_continual(model, classifier, data_loaders, device):
    """
    Evaluate continual learning performance across multiple tasks.

    Args:
        model (nn.Module): Feature extractor
        classifier (nn.Module): Linear classifier
        data_loaders (list): List of evaluation data loaders for each task
        device (torch.device): Device for evaluation

    Returns:
        metrics (dict): Evaluation metrics including forgetting
    """
    # Task-aware evaluation (with task ID)
    task_aware_accs = []
    # Task-agnostic evaluation (without task ID)
    task_agnostic_accs = []

    num_tasks = len(data_loaders)

    for task_id in range(num_tasks):
        # Task-aware evaluation
        metrics_aware = evaluate_linear(
            model, classifier, data_loaders[task_id], device, task_id=task_id
        )
        task_aware_accs.append(metrics_aware['acc1'])

        # Task-agnostic evaluation
        metrics_agnostic = evaluate_linear(
            model, classifier, data_loaders[task_id], device, task_id=None
        )
        task_agnostic_accs.append(metrics_agnostic['acc1'])

        print(f"Task {task_id} - Aware: {metrics_aware['acc1']:.2f}% - Agnostic: {metrics_agnostic['acc1']:.2f}%")

    # Calculate average linear precision (ALP)
    alp = np.mean([task_aware_accs, task_agnostic_accs])

    # Calculate overall accuracy
    acc_aware = np.mean(task_aware_accs)
    acc_agnostic = np.mean(task_agnostic_accs)

    print(f"Average Task-Aware Accuracy: {acc_aware:.2f}%")
    print(f"Average Task-Agnostic Accuracy: {acc_agnostic:.2f}%")
    print(f"Average Linear Precision (ALP): {alp:.2f}%")

    metrics = {
        'task_aware_accs': task_aware_accs,
        'task_agnostic_accs': task_agnostic_accs,
        'acc_aware': acc_aware,
        'acc_agnostic': acc_agnostic,
        'alp': alp
    }

    return metrics


def calculate_forgetting(accuracy_matrix):
    """
    Calculate forgetting metrics from accuracy matrix.

    Args:
        accuracy_matrix (np.ndarray): Matrix of accuracies [num_tasks, num_tasks]
            where accuracy_matrix[i, j] = accuracy on task j after learning task i

    Returns:
        metrics (dict): Forgetting metrics
    """
    num_tasks = accuracy_matrix.shape[0]

    # Calculate maximum accuracy for each task
    max_acc = np.max(accuracy_matrix, axis=0)

    # Calculate accuracy after all tasks
    final_acc = accuracy_matrix[-1, :]

    # Calculate forgetting for each task
    forgetting_per_task = max_acc - final_acc

    # Average forgetting (excluding the last task)
    mean_forgetting = np.mean(forgetting_per_task[:-1]) if num_tasks > 1 else 0.0

    # Calculate forward transfer
    # Assuming first row contains baseline performance
    forward_transfer = np.mean(final_acc - accuracy_matrix[0, :])

    metrics = {
        'forgetting_per_task': forgetting_per_task,
        'mean_forgetting': mean_forgetting,
        'forward_transfer': forward_transfer
    }

    print(f"Forgetting Analysis:")
    print(f"  Mean Forgetting: {mean_forgetting:.2f}%")
    print(f"  Forward Transfer: {forward_transfer:.2f}%")

    return metrics


def accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions.

    Args:
        output (torch.Tensor): Prediction outputs
        target (torch.Tensor): Target labels
        topk (tuple): Top-k values to compute accuracy

    Returns:
        res (list): List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def visualize_mask(model, images, device, mask_ratio=0.75, output_path=None):
    """
    Visualize the masks generated by S3UM.

    Args:
        model (nn.Module): CMUIM model
        images (torch.Tensor): [B, 3, H, W] input images
        device (torch.device): Device for evaluation
        mask_ratio (float): Fraction of patches to mask
        output_path (str, optional): Path to save visualizations
    """
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        # Get patch embeddings
        x = model.patch_embed(images)

        # Generate mask
        mask = model.masking_net.get_binary_mask(x, mask_ratio)

        # Reshape mask to match image patches
        patch_size = model.patch_size
        h = w = images.shape[-1] // patch_size

        # Create visualization
        from ..utils.visualization import visualize_masks
        visualize_masks(images, mask, h, w, patch_size, output_path)


def compare_representations(model_current, model_previous, images, device):
    """
    Compare feature representations between current and previous models.

    Args:
        model_current (nn.Module): Current model
        model_previous (nn.Module): Previous model
        images (torch.Tensor): [B, 3, H, W] input images
        device (torch.device): Device for evaluation

    Returns:
        metrics (dict): Similarity metrics
    """
    model_current.eval()
    model_previous.eval()

    images = images.to(device)

    with torch.no_grad():
        # Extract features
        features_current = model_current.forward_features(images)
        features_previous = model_previous.forward_features(images)

        # Normalize features
        features_current = F.normalize(features_current, dim=1)
        features_previous = F.normalize(features_previous, dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.mean(torch.sum(features_current * features_previous, dim=1))

        # Calculate L2 distance
        l2_dist = torch.mean(torch.norm(features_current - features_previous, dim=1))

    metrics = {
        'cosine_sim': cosine_sim.item(),
        'l2_dist': l2_dist.item()
    }

    print(f"Representation Comparison:")
    print(f"  Cosine Similarity: {cosine_sim.item():.4f}")
    print(f"  L2 Distance: {l2_dist.item():.4f}")

    return metrics