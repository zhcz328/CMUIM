"""
Miscellaneous utilities for S3UM architecture.

This module implements various utility functions used throughout the project.
"""

import os
import random
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path


class NativeScaler:
    """
    Gradient scaler for mixed precision training.

    Wraps PyTorch's GradScaler with additional functionality.
    """

    def __init__(self):
        """Initialize native PyTorch GradScaler."""
        self._scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def __call__(self, loss, optimizer, parameters=None, clip_grad=None,
                 clip_mode='norm', update_grad=True, create_graph=False):
        """
        Scale loss and perform backward pass.

        Args:
            loss: Computed loss
            optimizer: Optimizer to update parameters
            parameters: Model parameters (optional)
            clip_grad: Gradient clipping value (optional)
            clip_mode: Gradient clipping mode (norm or value)
            update_grad: Whether to update parameters
            create_graph: Whether to create graph for higher-order gradients
        """
        if self._scaler is not None:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    clip_gradients(parameters, clip_grad, mode=clip_mode)
                self._scaler.step(optimizer)
                self._scaler.update()
        else:
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    clip_gradients(parameters, clip_grad, mode=clip_mode)
                optimizer.step()

    def state_dict(self):
        """Get state dict for checkpointing."""
        return self._scaler.state_dict() if self._scaler else {}

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        if self._scaler and state_dict:
            self._scaler.load_state_dict(state_dict)


def clip_gradients(parameters, clip_grad, mode='norm'):
    """
    Clip gradients by norm or value.

    Args:
        parameters: Model parameters
        clip_grad: Clipping threshold
        mode: Clipping mode (norm or value)
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, clip_grad)
    else:
        raise ValueError(f"Unknown gradient clipping mode: {mode}")


def save_model(args, model, optimizer=None, loss_scaler=None, epoch=None,
               is_best=False, output_dir=None):
    """
    Save model checkpoint.

    Args:
        args: Command-line arguments
        model: Model to save
        optimizer: Optimizer state (optional)
        loss_scaler: Gradient scaler (optional)
        epoch: Current epoch (optional)
        is_best: Whether this is the best model (optional)
        output_dir: Directory to save checkpoint (optional)
    """
    if dist.get_rank() != the_get_rank():
        return

    # Use specified output directory or from args
    out_dir = output_dir if output_dir else args.output_dir

    # Create directory if not exists
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
    }

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if loss_scaler is not None:
        checkpoint['scaler'] = loss_scaler.state_dict()

    # Save checkpoint
    if is_best:
        save_path = os.path.join(out_dir, 'checkpoint-best.pth')
    else:
        save_path = os.path.join(out_dir, f'checkpoint-{epoch}.pth')

    torch.save(checkpoint, save_path)

    # Save latest checkpoint separately
    if epoch is not None:
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint-latest.pth'))


def load_model(args, model, optimizer=None, loss_scaler=None, strict=True):
    """
    Load model from checkpoint.

    Args:
        args: Command-line arguments
        model: Model to load weights into
        optimizer: Optimizer to load state (optional)
        loss_scaler: Gradient scaler to load state (optional)
        strict: Whether to strictly enforce that the keys in state_dict match
    """
    # Determine checkpoint path from args
    if args.resume:
        checkpoint_path = args.resume
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified, starting from scratch")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    msg = model.load_state_dict(checkpoint['model'], strict=strict)
    print(f"Loaded model with message: {msg}")

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded optimizer state")

    # Load scaler state if provided
    if loss_scaler is not None and 'scaler' in checkpoint:
        loss_scaler.load_state_dict(checkpoint['scaler'])
        print("Loaded gradient scaler state")

    # Return epoch from checkpoint if available
    return checkpoint.get('epoch', 0)


def the_get_rank():
    """
    Get process rank.

    Helper function to get the correct rank in distributed training.
    """
    return 0 if not dist.is_initialized() else dist.get_rank()


def setup_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False