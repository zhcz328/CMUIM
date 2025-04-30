#!/usr/bin/env python3
"""
Main entry point for S3UM (Selective State Space Ultrasound Masking).

This script provides a unified interface for training, testing, and visualization
of the S3UM architecture for continual self-supervised learning in ultrasound imaging.
"""

import os
import sys
import argparse
import datetime
import time
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config import get_args_parser, S3UMConfig
from models.model_cmuim import CMUIM
from models.sagp import SemanticAwareGradientPerturbation
from datasets.data_loader import create_dataset_loaders, create_continual_datasets
from engine.train import (
    train_one_epoch,
    train_one_epoch_continual,
    train_one_epoch_linear_probing,
    train_one_epoch_linear_probing_continual
)
from engine.test import (
    evaluate,
    evaluate_linear,
    evaluate_continual,
    calculate_forgetting,
    visualize_mask
)
from utils.misc import NativeScaler, save_model, load_model


def main(args):
    """
    Main function for S3UM training and evaluation.

    Args:
        args: Parsed command line arguments
    """
    # Create config from args
    config = S3UMConfig.from_args(args)
    print(f"Config: {config}")

    # Setup device
    device = torch.device(config.device)

    # Fix random seeds for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    # Create datasets and loaders for continual learning
    train_loaders, val_loaders, buffer_loaders, test_loaders = create_continual_datasets(
        config.data_path,
        config.n_tasks,
        config.batch_size,
        config.buffer_size,
        config.num_workers,
        seed=config.seed
    )

    # Select operation mode
    if args.mode == 'train':
        # Initial training on first task
        train_first_task(config, train_loaders[0], val_loaders[0], device, log_writer)

        # Continual learning on subsequent tasks
        if config.n_tasks > 1:
            train_continual(config, train_loaders, val_loaders, buffer_loaders, device, log_writer)

    elif args.mode == 'eval':
        # Evaluate trained model
        evaluate_model(config, test_loaders, device)

    elif args.mode == 'visualize':
        # Visualize model outputs
        visualize_model(config, test_loaders, device)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if log_writer is not None:
        log_writer.close()


def train_first_task(config, train_loader, val_loader, device, log_writer=None):
    """
    Train the model on the first task.

    Args:
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        log_writer: TensorBoard writer
    """
    print(f"Training on first task...")

    # Create model
    model = CMUIM(
        img_size=config.input_size,
        patch_size=config.patch_size,
        in_chans=3,
        embed_dim=config.embed_dim,
        depth=config.depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        lambda_area=config.lambda_area,
        lambda_diversity=config.lambda_div,
        masking_depth=5,
        dropout=config.dropout
    )
    model.to(device)

    # Create optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)

    # Create loss scalers for mixed precision training
    loss_scaler = NativeScaler()
    loss_scaler_mask = NativeScaler()

    # Load checkpoint if available
    if args.resume:
        load_model(args, model, optimizer, loss_scaler)

    print(f"Start training for {config.epochs} epochs")
    start_time = time.time()

    min_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.epochs):
        # Calculate curriculum factor for mask training
        if epoch < (config.epochs / 2):
            curriculum_factor = 1 - (2 * epoch) / config.epochs
        else:
            curriculum_factor = 1 - (2 * (epoch + 1)) / config.epochs

        # Train for one epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            curriculum_factor, loss_scaler, loss_scaler_mask,
            log_writer=log_writer, args=args
        )

        # Evaluate
        eval_stats = evaluate(model, val_loader, device)
        val_loss = eval_stats['recon_loss']

        print(f"Epoch {epoch} - Validation loss: {val_loss:.4f}")

        # Save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            save_model(
                args=args,
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=best_epoch,
                is_best=True
            )

        # Regular checkpointing
        if (epoch + 1) % config.save_freq == 0 or epoch + 1 == config.epochs:
            save_model(
                args=args,
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch
            )

    # Linear evaluation
    train_linear_classifier(config, model, train_loader, val_loader, device, log_writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"First task training time: {total_time_str}")


def train_continual(config, train_loaders, val_loaders, buffer_loaders, device, log_writer=None):
    """
    Train the model continually on multiple tasks.

    Args:
        config: Configuration object
        train_loaders: List of training data loaders
        val_loaders: List of validation data loaders
        buffer_loaders: List of buffer data loaders
        device: Computation device
        log_writer: TensorBoard writer
    """
    print(f"Starting continual learning for {config.n_tasks} tasks...")

    # Initialize accuracy matrix for forgetting calculation
    accuracy_matrix = np.zeros((config.n_tasks, config.n_tasks))

    # SAGP module for buffer samples
    sagp = SemanticAwareGradientPerturbation(eta=config.beta)

    # For each task (starting from the second)
    for task_id in range(1, config.n_tasks):
        print(f"Training on task {task_id}...")

        # Load previous model
        model_previous = CMUIM(
            img_size=config.input_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
            depth=config.depth,
            d_state=config.d_state,
            d_conv=config.d_conv,
            lambda_area=config.lambda_area,
            lambda_diversity=config.lambda_div,
            masking_depth=5,
            dropout=config.dropout
        )

        # Load best model from previous task
        model_previous.to(device)
        checkpoint_path = Path(args.output_dir) / "checkpoint-best.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_previous.load_state_dict(checkpoint['model'])
        model_previous.eval()  # Set to evaluation mode

        # Create new model initialized with previous model's weights
        model_current = CMUIM(
            img_size=config.input_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
            depth=config.depth,
            d_state=config.d_state,
            d_conv=config.d_conv,
            lambda_area=config.lambda_area,
            lambda_diversity=config.lambda_div,
            masking_depth=5,
            dropout=config.dropout
        )
        model_current.to(device)
        model_current.load_state_dict(checkpoint['model'])

        # Create optimizer for current model
        parameters = filter(lambda p: p.requires_grad, model_current.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)

        # Create loss scalers for mixed precision training
        loss_scaler = NativeScaler()
        loss_scaler_mask = NativeScaler()

        # Train on current task with contrastive alignment to previous model
        print(f"Start training for task {task_id} for {config.epochs} epochs")
        start_time = time.time()

        min_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(config.epochs):
            # Calculate curriculum factor
            if epoch < (config.epochs / 2):
                curriculum_factor = 1 - (2 * epoch) / config.epochs
            else:
                curriculum_factor = 1 - (2 * (epoch + 1)) / config.epochs

            # Train for one epoch with continual learning
            train_stats = train_one_epoch_continual(
                model_current, model_previous,
                train_loaders[task_id], buffer_loaders[task_id],
                optimizer, device, epoch,
                curriculum_factor, loss_scaler, loss_scaler_mask,
                sagp=sagp, log_writer=log_writer, args=args
            )

            # Evaluate on current task
            eval_stats = evaluate(model_current, val_loaders[task_id], device)
            val_loss = eval_stats['recon_loss']

            print(f"Epoch {epoch} - Task {task_id} - Validation loss: {val_loss:.4f}")

            # Save best model
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                # Save model to task-specific directory
                task_output_dir = Path(args.output_dir).parent / f"task_{task_id}"
                task_output_dir.mkdir(parents=True, exist_ok=True)
                save_model(
                    args=args,
                    model=model_current,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=best_epoch,
                    is_best=True,
                    output_dir=str(task_output_dir)
                )

        # Linear evaluation
        classifier = train_linear_classifier_continual(
            config, model_current, model_previous,
            train_loaders[:task_id + 1], val_loaders[:task_id + 1],
            device, log_writer, task_id
        )

        # Evaluate on all tasks seen so far
        for eval_task_id in range(task_id + 1):
            metrics = evaluate_linear(
                model_current, classifier, val_loaders[eval_task_id], device
            )
            accuracy_matrix[task_id, eval_task_id] = metrics['acc1']

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Task {task_id} training time: {total_time_str}")

    # Calculate forgetting metrics
    forgetting_metrics = calculate_forgetting(accuracy_matrix)
    print(f"Final forgetting metrics: {forgetting_metrics}")

    # Save accuracy matrix
    np.save(Path(args.output_dir) / "accuracy_matrix.npy", accuracy_matrix)


def train_linear_classifier(config, model, train_loader, val_loader, device, log_writer=None):
    """
    Train a linear classifier on top of the frozen encoder.

    Args:
        config: Configuration object
        model: Pretrained model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        log_writer: TensorBoard writer

    Returns:
        classifier: Trained linear classifier
    """
    print("Training linear classifier...")

    # Create linear classifier
    classifier = torch.nn.Linear(config.embed_dim, get_num_classes(train_loader))
    classifier.to(device)

    # Freeze encoder
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Train linear classifier
    best_acc = 0.0
    for epoch in range(100):  # Fixed number of epochs for linear evaluation
        train_one_epoch_linear_probing(
            model, classifier, criterion, train_loader,
            optimizer, device, epoch, args=args, log_writer=log_writer
        )

        metrics = evaluate_linear(model, classifier, val_loader, device)
        acc = metrics['acc1']

        print(f"Linear Epoch {epoch} - Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            # Save best classifier
            torch.save(
                {
                    'model': classifier.state_dict(),
                    'epoch': epoch,
                    'acc': acc
                },
                Path(args.output_dir) / "classifier-best.pth"
            )

    print(f"Best linear classifier accuracy: {best_acc:.2f}%")
    return classifier


def train_linear_classifier_continual(
        config, model_current, model_previous, train_loaders, val_loaders, device, log_writer=None, task_id=None
):
    """
    Train a linear classifier with knowledge distillation for continual learning.

    Args:
        config: Configuration object
        model_current: Current task model
        model_previous: Previous task model
        train_loaders: List of training data loaders for all tasks
        val_loaders: List of validation data loaders for all tasks
        device: Computation device
        log_writer: TensorBoard writer
        task_id: Current task ID

    Returns:
        classifier: Trained linear classifier
    """
    print(f"Training linear classifier for task {task_id} with knowledge distillation...")

    # Create combined data loader for all tasks seen so far
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
    combined_loader = torch.utils.data.DataLoader(
        combined_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )

    # Get total number of classes
    num_classes = get_num_classes(combined_loader)

    # Create linear classifier
    classifier = torch.nn.Linear(config.embed_dim, num_classes)
    classifier.to(device)

    # Load previous classifier if available
    if task_id > 0:
        # Load previous classifier
        prev_classifier = torch.nn.Linear(config.embed_dim, get_num_classes(train_loaders[0]))
        prev_classifier.to(device)

        # Load weights from previous task
        prev_classifier_path = Path(args.output_dir).parent / f"task_{task_id - 1}_linear" / "classifier-best.pth"
        if prev_classifier_path.exists():
            checkpoint = torch.load(prev_classifier_path, map_location='cpu')
            prev_classifier.load_state_dict(checkpoint['model'])

            # Initialize current classifier with previous weights for shared classes
            with torch.no_grad():
                classifier.weight.data[:prev_classifier.weight.shape[0]] = prev_classifier.weight.data
                classifier.bias.data[:prev_classifier.bias.shape[0]] = prev_classifier.bias.data

        prev_classifier.eval()  # Set to evaluation mode
    else:
        prev_classifier = None

    # Freeze encoder
    model_current.eval()
    for param in model_current.parameters():
        param.requires_grad = False

    if model_previous is not None:
        model_previous.eval()
        for param in model_previous.parameters():
            param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Train linear classifier
    best_acc = 0.0
    for epoch in range(100):  # Fixed number of epochs for linear evaluation
        if task_id > 0 and prev_classifier is not None:
            # Train with knowledge distillation
            train_one_epoch_linear_probing_continual(
                model_current, classifier, prev_classifier, criterion,
                combined_loader, optimizer, device, epoch,
                alpha=config.alpha, temperature=2.0,
                args=args, log_writer=log_writer
            )
        else:
            # Standard training for first task
            train_one_epoch_linear_probing(
                model_current, classifier, criterion,
                combined_loader, optimizer, device, epoch,
                args=args, log_writer=log_writer
            )

        # Evaluate on validation set of current task
        metrics = evaluate_linear(model_current, classifier, val_loaders[-1], device)
        acc = metrics['acc1']

        print(f"Linear Epoch {epoch} - Task {task_id} - Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            # Save best classifier to task-specific directory
            task_output_dir = Path(args.output_dir).parent / f"task_{task_id}_linear"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'model': classifier.state_dict(),
                    'epoch': epoch,
                    'acc': acc
                },
                task_output_dir / "classifier-best.pth"
            )

    print(f"Best linear classifier accuracy for task {task_id}: {best_acc:.2f}%")
    return classifier


def evaluate_model(config, test_loaders, device):
    """
    Evaluate trained model on test data.

    Args:
        config: Configuration object
        test_loaders: List of test data loaders
        device: Computation device
    """
    print("Evaluating model...")

    # Load model and classifier for each task
    models = []
    classifiers = []

    for task_id in range(config.n_tasks):
        # Load model
        model = CMUIM(
            img_size=config.input_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
            depth=config.depth,
            d_state=config.d_state,
            d_conv=config.d_conv,
            lambda_area=config.lambda_area,
            lambda_diversity=config.lambda_div,
            masking_depth=5,
            dropout=config.dropout
        )
        model.to(device)

        # Load weights
        model_path = Path(args.output_dir).parent / f"task_{task_id}" / "checkpoint-best.pth"
        if task_id == 0:
            model_path = Path(args.output_dir) / "checkpoint-best.pth"

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.eval()
            models.append(model)

            # Load classifier
            classifier = torch.nn.Linear(config.embed_dim, get_num_classes(test_loaders[task_id]))
            classifier.to(device)

            classifier_path = Path(args.output_dir).parent / f"task_{task_id}_linear" / "classifier-best.pth"
            if task_id == 0:
                classifier_path = Path(args.output_dir) / "classifier-best.pth"

            if classifier_path.exists():
                checkpoint = torch.load(classifier_path, map_location='cpu')
                classifier.load_state_dict(checkpoint['model'])
                classifier.eval()
                classifiers.append(classifier)

    # Evaluate continual learning performance
    if len(models) > 1 and len(classifiers) > 1:
        # Use the final model and classifier for evaluation
        final_model = models[-1]
        final_classifier = classifiers[-1]

        metrics = evaluate_continual(final_model, final_classifier, test_loaders, device)

        print(f"Continual Learning Evaluation:")
        print(f"  Task-Aware Accuracy: {metrics['acc_aware']:.2f}%")
        print(f"  Task-Agnostic Accuracy: {metrics['acc_agnostic']:.2f}%")
        print(f"  Average Linear Precision (ALP): {metrics['alp']:.2f}%")
    else:
        # Single task evaluation
        for task_id, (model, classifier, loader) in enumerate(zip(models, classifiers, test_loaders)):
            metrics = evaluate_linear(model, classifier, loader, device)

            print(f"Task {task_id} Evaluation:")
            print(f"  Accuracy: {metrics['acc1']:.2f}%")


def visualize_model(config, test_loaders, device):
    """
    Visualize model outputs and masks.

    Args:
        config: Configuration object
        test_loaders: List of test data loaders
        device: Computation device
    """
    print("Visualizing model outputs...")

    # Load final model
    model = CMUIM(
        img_size=config.input_size,
        patch_size=config.patch_size,
        in_chans=3,
        embed_dim=config.embed_dim,
        depth=config.depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        lambda_area=config.lambda_area,
        lambda_diversity=config.lambda_div,
        masking_depth=5,
        dropout=config.dropout
    )
    model.to(device)

    # Find the latest task model
    latest_task = config.n_tasks - 1
    while latest_task >= 0:
        model_path = Path(args.output_dir).parent / f"task_{latest_task}" / "checkpoint-best.pth"
        if latest_task == 0:
            model_path = Path(args.output_dir) / "checkpoint-best.pth"

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.eval()
            break

        latest_task -= 1

    if latest_task < 0:
        print("No trained model found. Exiting visualization.")
        return

    # Create visualization directory
    vis_dir = Path(args.output_dir).parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Visualize masks for samples from each task
    for task_id, loader in enumerate(test_loaders):
        # Get a batch of images
        images, _ = next(iter(loader))
        images = images[:4].to(device)  # Visualize only a few images

        # Visualize masks
        output_path = vis_dir / f"task_{task_id}_masks"
        output_path.mkdir(parents=True, exist_ok=True)

        visualize_mask(
            model, images, device,
            mask_ratio=config.mask_ratio,
            output_path=str(output_path)
        )

        print(f"Visualizations for task {task_id} saved to {output_path}")


def get_num_classes(data_loader):
    """
    Get the number of classes in a data loader.

    Args:
        data_loader: Data loader

    Returns:
        num_classes: Number of classes
    """
    if hasattr(data_loader.dataset, 'classes'):
        return len(data_loader.dataset.classes)
    elif hasattr(data_loader.dataset, 'dataset') and hasattr(data_loader.dataset.dataset, 'classes'):
        return len(data_loader.dataset.dataset.classes)
    elif hasattr(data_loader.dataset, 'tensors'):
        # For TensorDataset, assume target is the second tensor
        return len(torch.unique(data_loader.dataset.tensors[1]))
    else:
        # Try to infer from the data
        for _, targets in data_loader:
            return len(torch.unique(targets))
        return 0


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)