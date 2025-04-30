"""
Training engine for S3UM architecture.

This module implements the training procedures for the Continual Masked
Ultrasound Image Modeling framework, including bi-level optimization
and continual learning mechanisms.
"""

import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.metrics import AverageMeter


def cal_contrastive_loss(img1_rep, img2_rep, temperature=0.05, bidirect_contrast=True):
    """
    Calculate contrastive loss between two sets of representations.

    Args:
        img1_rep (torch.Tensor): First set of representations
        img2_rep (torch.Tensor): Second set of representations
        temperature (float): Temperature parameter for softmax
        bidirect_contrast (bool): Whether to use bidirectional contrast

    Returns:
        loss (torch.Tensor): Contrastive loss
        accuracy (torch.Tensor): Contrastive accuracy
    """
    # Normalize representations
    img1_rep = F.normalize(img1_rep, dim=-1)
    img2_rep = F.normalize(img2_rep, dim=-1)

    # Reshape if needed
    img1_rep = img1_rep.view(img1_rep.size()[0], -1)
    img2_rep = img2_rep.view(img2_rep.size()[0], -1)

    # Compute similarity matrix
    total = torch.mm(img1_rep, torch.transpose(img2_rep, 0, 1)) / temperature

    # Calculate contrastive loss
    if not bidirect_contrast:
        # Single direction contrast
        nce = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
        c_acc = torch.sum(torch.eq(torch.argmax(F.softmax(total, dim=0), dim=0),
                                  torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]
        return nce, c_acc
    else:
        # Bidirectional contrast
        nce_1 = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
        nce_2 = -torch.mean(torch.diag(F.log_softmax(total.t(), dim=0)))

        c_acc_1 = torch.sum(torch.eq(torch.argmax(F.softmax(total, dim=0), dim=0),
                                    torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]
        c_acc_2 = torch.sum(torch.eq(torch.argmax(F.softmax(total.t(), dim=0), dim=0),
                                    torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]

        nce = (nce_1 + nce_2) / 2
        c_acc = (c_acc_1 + c_acc_2) / 2

        return nce, c_acc


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    curriculum_factor,
    loss_scaler,
    loss_scaler_mask,
    args=None,
    log_writer=None,
):
    """
    Train the CMUIM model for one epoch using the bi-level optimization strategy.

    This function implements the bi-level optimization strategy as described in the paper:
    1. Stage I - Anatomical Mask Meta-Optimization
    2. Stage II - Task Representation via Contrastive Alignment

    Args:
        model (nn.Module): CMUIM model
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for training
        epoch (int): Current epoch
        curriculum_factor (float): Factor for curriculum learning
        loss_scaler (NativeScaler): Gradient scaler for Stage II
        loss_scaler_mask (NativeScaler): Gradient scaler for Stage I
        args (argparse.Namespace): Training arguments
        log_writer (SummaryWriter): TensorBoard writer

    Returns:
        avg_loss (float): Average loss for the epoch
    """
    model.train()
    model = model.float()

    print(f"Epoch: [{epoch}]")
    print_freq = 20

    # Initialize metrics
    stage1_losses = AverageMeter("Stage I Loss")
    stage2_losses = AverageMeter("Stage II Loss")
    recon_losses = AverageMeter("Reconstruction Loss")
    align_losses = AverageMeter("Alignment Loss")
    area_losses = AverageMeter("Area Loss")
    diversity_losses = AverageMeter("Diversity Loss")

    num_batches = len(data_loader)

    for data_iter_step, (samples, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Adjust learning rate per iteration
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # ---- Stage I: Anatomical Mask Meta-Optimization ----
        # Train the masking network with feature representation regularization
        model.freeze_backbone()
        model.unfreeze_maskingnet()

        with torch.cuda.amp.autocast():
            # Using the mask training mode to get all losses
            loss_recon, loss_area, loss_kl, loss_diversity = model(
                samples,
                mask_ratio=args.mask_ratio,
                train_mask=True
            )

            # Combine Stage I losses as per the paper
            # 𝓛_Stage I = 𝔼_{x∼D_k}[λ_ssl·𝓛_ssl + 𝓛_area + 𝓛_Div]
            stage1_loss = curriculum_factor * loss_recon + loss_area + loss_diversity

        # Update metrics
        stage1_losses.update(stage1_loss.item())
        area_losses.update(loss_area.item())
        diversity_losses.update(loss_diversity.item())

        # Gradient update for Stage I
        optimizer.zero_grad()
        loss_scaler_mask(
            stage1_loss,
            optimizer,
            parameters=filter(lambda p: p.requires_grad, model.parameters()),
            update_grad=True
        )

        # Reset model state for Stage II
        model.unfreeze_backbone()

        # ---- Stage II: Task Representation via Contrastive Alignment ----
        # Now train the encoder network while keeping masking network frozen
        model.freeze_maskingnet()

        with torch.cuda.amp.autocast():
            # Current task reconstruction loss
            loss_recon, latent_current = model(samples, mask_ratio=args.mask_ratio)
            recon_losses.update(loss_recon.item())

            # Cross-task alignment if there is a previous model
            if hasattr(model, 'previous_encoder') and model.previous_encoder is not None:
                # Get features from both current and previous encoders
                with torch.no_grad():
                    _, latent_previous = model.previous_encoder(samples, mask_ratio=args.mask_ratio)

                # Calculate contrastive alignment loss
                align_loss, _ = cal_contrastive_loss(
                    latent_current,
                    latent_previous,
                    temperature=args.temperature if hasattr(args, 'temperature') else 0.05
                )
                align_losses.update(align_loss.item())

                # Stage II loss: Task-Specific Reconstruction + Cross-Task Consistency
                # 𝓛_Stage II = 𝔼_{x∼𝓓_k}[𝓛_ssl(x;θ)] + 𝔼_{x∼𝓓_k ∪ 𝓓_k_Inc}[𝓛_align(x;θ,θ⁻)]
                stage2_loss = loss_recon + args.alpha * align_loss if hasattr(args, 'alpha') else loss_recon + align_loss
            else:
                # For the first task, no alignment loss
                stage2_loss = loss_recon
                align_losses.update(0.0)

        stage2_losses.update(stage2_loss.item())

        # Gradient update for Stage II
        optimizer.zero_grad()
        loss_scaler(
            stage2_loss,
            optimizer,
            parameters=filter(lambda p: p.requires_grad, model.parameters()),
            update_grad=True
        )

        # Reset model state
        model.unfreeze_maskingnet()

        # Log metrics
        if (data_iter_step + 1) % print_freq == 0:
            print(f"Step: [{data_iter_step+1}/{num_batches}] "
                  f"Stage I Loss: {stage1_losses.avg:.4f} "
                  f"Stage II Loss: {stage2_losses.avg:.4f} "
                  f"Recon Loss: {recon_losses.avg:.4f} "
                  f"Align Loss: {align_losses.avg:.4f} "
                  f"Area Loss: {area_losses.avg:.4f} "
                  f"Diversity Loss: {diversity_losses.avg:.4f}")

            if log_writer is not None:
                epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
                log_writer.add_scalar('train/stage1_loss', stage1_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/stage2_loss', stage2_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/recon_loss', recon_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/align_loss', align_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/area_loss', area_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/diversity_loss', diversity_losses.avg, epoch_1000x)
                log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch_1000x)

    # Return the average of both stage losses
    total_avg_loss = (stage1_losses.avg + stage2_losses.avg) / 2
    return total_avg_loss


def train_one_epoch_continual(
    model_current,
    model_previous,
    data_loader,
    buffer_loader,
    optimizer,
    device,
    epoch,
    curriculum_factor,
    loss_scaler,
    loss_scaler_mask,
    sagp,
    args=None,
    log_writer=None,
):
    """
    Train the CMUIM model with continual learning for one epoch.

    This function implements the continual learning strategy with:
    1. Stage I - Anatomical Mask Meta-Optimization with buffer samples
    2. Stage II - Task Representation via Contrastive Alignment with SAGP

    Args:
        model_current (nn.Module): Current CMUIM model
        model_previous (nn.Module): Frozen previous CMUIM model
        data_loader (DataLoader): Current task data loader
        buffer_loader (DataLoader): Memory buffer data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for training
        epoch (int): Current epoch
        curriculum_factor (float): Factor for curriculum learning
        loss_scaler (NativeScaler): Gradient scaler for Stage II
        loss_scaler_mask (NativeScaler): Gradient scaler for Stage I
        sagp (SemanticAwareGradientPerturbation): SAGP module
        args (argparse.Namespace): Training arguments
        log_writer (SummaryWriter): TensorBoard writer

    Returns:
        metrics (dict): Training metrics for the epoch
    """
    model_current.train()
    model_previous.eval()

    # Convert models to float precision
    model_current = model_current.float()
    model_previous = model_previous.float()

    print(f"Continual Learning Epoch: [{epoch}]")
    print_freq = 20

    # Initialize metrics
    stage1_losses = AverageMeter("Stage I Loss")
    stage2_losses = AverageMeter("Stage II Loss")
    recon_losses = AverageMeter("Reconstruction Loss")
    align_losses = AverageMeter("Alignment Loss")
    contra_acc = AverageMeter("Contrastive Accuracy")
    area_losses = AverageMeter("Area Loss")
    diversity_losses = AverageMeter("Diversity Loss")

    num_batches = len(data_loader)
    buffer_iter = iter(buffer_loader)

    for data_iter_step, (samples, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Adjust learning rate per iteration
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # Get buffer samples - cycle if buffer is smaller than task data
        try:
            buffer_samples, _ = next(buffer_iter)
        except StopIteration:
            buffer_iter = iter(buffer_loader)
            buffer_samples, _ = next(buffer_iter)

        buffer_samples = buffer_samples.to(device, non_blocking=True)

        # Apply SAGP perturbation to buffer samples
        if sagp is not None:
            with torch.no_grad():
                perturbed_buffer = sagp.generate_perturbations(
                    model_current, model_previous, buffer_samples, device
                )
        else:
            perturbed_buffer = buffer_samples

        # ---- Stage I: Anatomical Mask Meta-Optimization ----
        model_current.freeze_backbone()
        model_current.unfreeze_maskingnet()

        # Combine current samples and buffer samples for diverse training
        combined_samples = torch.cat([
            samples[:samples.size(0)//2],
            perturbed_buffer[:perturbed_buffer.size(0)//2]
        ], dim=0)

        with torch.cuda.amp.autocast():
            # Train masking network on combined samples
            loss_recon, loss_area, loss_kl, loss_diversity = model_current(
                combined_samples,
                mask_ratio=args.mask_ratio,
                train_mask=True
            )

            # Stage I loss
            stage1_loss = curriculum_factor * loss_recon + loss_area + loss_diversity

        # Update metrics
        stage1_losses.update(stage1_loss.item())
        area_losses.update(loss_area.item())
        diversity_losses.update(loss_diversity.item())

        # Gradient update for Stage I
        optimizer.zero_grad()
        loss_scaler_mask(
            stage1_loss,
            optimizer,
            parameters=filter(lambda p: p.requires_grad, model_current.parameters()),
            update_grad=True
        )

        # Reset model state for Stage II
        model_current.unfreeze_backbone()

        # ---- Stage II: Task Representation via Contrastive Alignment ----
        model_current.freeze_maskingnet()

        with torch.cuda.amp.autocast():
            # Current task reconstruction
            loss_recon, latent_current = model_current(samples, mask_ratio=args.mask_ratio)
            recon_losses.update(loss_recon.item())

            # Cross-task alignment with buffer samples
            _, latent_current_buffer = model_current(perturbed_buffer, mask_ratio=args.mask_ratio)

            with torch.no_grad():
                _, latent_previous_buffer = model_previous(perturbed_buffer, mask_ratio=args.mask_ratio)

            # Calculate contrastive alignment loss
            contra_loss, acc = cal_contrastive_loss(
                latent_current_buffer,
                latent_previous_buffer,
                temperature=args.temperature if hasattr(args, 'temperature') else 0.05
            )

            # Stage II loss: Task-Specific Reconstruction + Cross-Task Consistency
            stage2_loss = loss_recon + (args.alpha if hasattr(args, 'alpha') else 0.5) * contra_loss

        # Update metrics
        stage2_losses.update(stage2_loss.item())
        align_losses.update(contra_loss.item())
        contra_acc.update(acc.item())

        # Gradient update for Stage II
        optimizer.zero_grad()
        loss_scaler(
            stage2_loss,
            optimizer,
            parameters=filter(lambda p: p.requires_grad, model_current.parameters()),
            update_grad=True
        )

        # Reset model state
        model_current.unfreeze_maskingnet()

        # Log metrics
        if (data_iter_step + 1) % print_freq == 0:
            print(f"Step: [{data_iter_step+1}/{num_batches}] "
                  f"Stage I Loss: {stage1_losses.avg:.4f} "
                  f"Stage II Loss: {stage2_losses.avg:.4f} "
                  f"Recon Loss: {recon_losses.avg:.4f} "
                  f"Align Loss: {align_losses.avg:.4f} "
                  f"Contra Acc: {contra_acc.avg:.4f}")

            if log_writer is not None:
                epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
                log_writer.add_scalar('train_cl/stage1_loss', stage1_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/stage2_loss', stage2_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/recon_loss', recon_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/align_loss', align_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/contra_acc', contra_acc.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/area_loss', area_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/diversity_loss', diversity_losses.avg, epoch_1000x)
                log_writer.add_scalar('train_cl/lr', optimizer.param_groups[0]['lr'], epoch_1000x)

    # Return all metrics for logging
    metrics = {
        'stage1_loss': stage1_losses.avg,
        'stage2_loss': stage2_losses.avg,
        'recon_loss': recon_losses.avg,
        'align_loss': align_losses.avg,
        'contra_acc': contra_acc.avg,
        'area_loss': area_losses.avg,
        'diversity_loss': diversity_losses.avg,
    }

    return metrics


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with cosine schedule.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (float): Current epoch (can be fractional)
        args (argparse.Namespace): Training arguments
    """
    if epoch < args.warmup_epochs:
        # Linear warmup
        lr = args.lr * epoch / args.warmup_epochs
    else:
        # Cosine decay
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch_linear_probing(
    model,
    classifier,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler=None,
    clip_grad=None,
    args=None,
    log_writer=None,
):
    """
    Train a linear classifier on top of frozen features.

    Args:
        model (nn.Module): Feature extractor (frozen)
        classifier (nn.Module): Linear classifier
        criterion (nn.Module): Loss function
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for training
        epoch (int): Current epoch
        loss_scaler (NativeScaler): Gradient scaler
        clip_grad (float): Gradient clipping value
        args (argparse.Namespace): Training arguments
        log_writer (SummaryWriter): TensorBoard writer

    Returns:
        top1_accuracy (float): Top-1 accuracy
    """
    model.eval()  # Freeze the feature extractor
    classifier.train()

    print(f"Linear Probing Epoch: [{epoch}]")
    print_freq = 20

    # Initialize metrics
    losses = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    num_batches = len(data_loader)

    for data_iter_step, (samples, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Adjust learning rate per iteration
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Extract features
        with torch.no_grad():
            features = model.forward_features(samples)

        # Forward pass through classifier
        outputs = classifier(features)
        loss = criterion(outputs, targets)

        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), samples.size(0))
        top1.update(acc1.item(), samples.size(0))
        top5.update(acc5.item(), samples.size(0))

        # Backward and optimize
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=classifier.parameters(),
                update_grad=True
            )
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad)
            optimizer.step()

        # Log metrics
        if (data_iter_step + 1) % print_freq == 0:
            print(f"Step: [{data_iter_step+1}/{num_batches}] "
                  f"Loss: {losses.avg:.4f} "
                  f"Acc@1: {top1.avg:.4f} "
                  f"Acc@5: {top5.avg:.4f}")

            if log_writer is not None:
                epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
                log_writer.add_scalar('linprobe/loss', losses.avg, epoch_1000x)
                log_writer.add_scalar('linprobe/acc1', top1.avg, epoch_1000x)
                log_writer.add_scalar('linprobe/acc5', top5.avg, epoch_1000x)
                log_writer.add_scalar('linprobe/lr', optimizer.param_groups[0]['lr'], epoch_1000x)

    return top1.avg


def train_one_epoch_linear_probing_continual(
    model,
    classifier,
    old_classifier,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    alpha=0.8,
    temperature=2.0,
    loss_scaler=None,
    clip_grad=None,
    args=None,
    log_writer=None,
):
    """
    Train a linear classifier with knowledge distillation from previous classifier.

    Args:
        model (nn.Module): Feature extractor (frozen)
        classifier (nn.Module): Current linear classifier
        old_classifier (nn.Module): Previous linear classifier
        criterion (nn.Module): Loss function
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for training
        epoch (int): Current epoch
        alpha (float): Weight for balancing CE and distillation loss
        temperature (float): Temperature for knowledge distillation
        loss_scaler (NativeScaler): Gradient scaler
        clip_grad (float): Gradient clipping value
        args (argparse.Namespace): Training arguments
        log_writer (SummaryWriter): TensorBoard writer

    Returns:
        top1_accuracy (float): Top-1 accuracy
    """
    model.eval()  # Freeze the feature extractor
    classifier.train()
    old_classifier.eval()  # Freeze the old classifier

    print(f"Linear Probing Continual Epoch: [{epoch}]")
    print_freq = 20

    # Initialize metrics
    losses = AverageMeter("Loss")
    ce_losses = AverageMeter("CE Loss")
    kd_losses = AverageMeter("KD Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    num_batches = len(data_loader)

    for data_iter_step, (samples, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Adjust learning rate per iteration
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Extract features
        with torch.no_grad():
            features = model.forward_features(samples)

        # Forward pass through current classifier
        outputs = classifier(features)

        # Classification loss
        ce_loss = criterion(outputs, targets)

        # Knowledge distillation loss
        with torch.no_grad():
            old_outputs = old_classifier(features)

        # Compute soft targets
        soft_targets = F.softmax(old_outputs / temperature, dim=1)
        outputs_soft = F.log_softmax(outputs / temperature, dim=1)

        # KL divergence loss
        kd_loss = F.kl_div(outputs_soft, soft_targets, reduction='batchmean') * (temperature**2)

        # Combined loss
        loss = alpha * ce_loss + (1 - alpha) * kd_loss

        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), samples.size(0))
        ce_losses.update(ce_loss.item(), samples.size(0))
        kd_losses.update(kd_loss.item(), samples.size(0))
        top1.update(acc1.item(), samples.size(0))
        top5.update(acc5.item(), samples.size(0))

        # Backward and optimize
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=classifier.parameters(),
                update_grad=True
            )
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip_grad)
            optimizer.step()

        # Log metrics
        if (data_iter_step + 1) % print_freq == 0:
            print(f"Step: [{data_iter_step+1}/{num_batches}] "
                  f"Loss: {losses.avg:.4f} "
                  f"CE: {ce_losses.avg:.4f} "
                  f"KD: {kd_losses.avg:.4f} "
                  f"Acc@1: {top1.avg:.4f}")

            if log_writer is not None:
                epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
                log_writer.add_scalar('linprobe_cl/loss', losses.avg, epoch_1000x)
                log_writer.add_scalar('linprobe_cl/ce_loss', ce_losses.avg, epoch_1000x)
                log_writer.add_scalar('linprobe_cl/kd_loss', kd_losses.avg, epoch_1000x)
                log_writer.add_scalar('linprobe_cl/acc1', top1.avg, epoch_1000x)
                log_writer.add_scalar('linprobe_cl/acc5', top5.avg, epoch_1000x)
                log_writer.add_scalar('linprobe_cl/lr', optimizer.param_groups[0]['lr'], epoch_1000x)

    return top1.avg


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