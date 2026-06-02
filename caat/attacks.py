"""
attacks.py — Adversarial attack implementations for evaluation.

FGSM and PGD (Madry et al. 2018) attacks.
Used in evaluation only — CAAT has its own inner-loop PGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon, mean, std):
    """
    Fast Gradient Sign Method (Goodfellow et al. 2015).

    Args:
        model: classifier
        images: (B, C, H, W) normalized images
        labels: (B,) true labels
        epsilon: perturbation budget (in [0,1] pixel space)
        mean: normalization mean (tuple)
        std: normalization std (tuple)

    Returns:
        adv_images: (B, C, H, W) adversarial images (normalized)
        success: (B,) boolean mask of successful attacks
    """
    device = images.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    # Convert epsilon to normalized space per channel
    eps_norm = epsilon / std_t

    images_req = images.detach().clone().requires_grad_(True)
    logits = model(images_req)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    # FGSM step
    perturbation = eps_norm * images_req.grad.sign()
    adv_images = images_req.detach() + perturbation

    # Clamp to valid normalized range
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t
    adv_images = torch.max(torch.min(adv_images, upper), lower)

    # Check success
    with torch.no_grad():
        adv_logits = model(adv_images)
        adv_preds = adv_logits.argmax(dim=1)
        success = (adv_preds != labels)

    return adv_images, success


def pgd_attack(model, images, labels, epsilon, alpha, steps, mean, std,
               random_start=True):
    """
    Projected Gradient Descent (Madry et al. 2018).

    Args:
        model: classifier
        images: (B, C, H, W) normalized images
        labels: (B,) true labels
        epsilon: perturbation budget (in [0,1] pixel space)
        alpha: step size (in [0,1] pixel space)
        steps: number of PGD steps
        mean: normalization mean (tuple)
        std: normalization std (tuple)
        random_start: initialize delta randomly within epsilon ball

    Returns:
        adv_images: (B, C, H, W) adversarial images (normalized)
        success: (B,) boolean mask of successful attacks
    """
    device = images.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    eps_norm = epsilon / std_t
    alpha_norm = alpha / std_t

    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    # Initialize perturbation
    if random_start:
        delta = torch.zeros_like(images).uniform_(-1, 1) * eps_norm
        delta = torch.max(torch.min(images + delta, upper), lower) - images
    else:
        delta = torch.zeros_like(images)

    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(images + delta)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        grad = delta.grad.detach()
        delta = delta.detach() + alpha_norm * grad.sign()

        # Project back to epsilon ball
        delta = torch.max(torch.min(delta, eps_norm), -eps_norm)

        # Ensure valid image range
        delta = torch.max(torch.min(images + delta, upper), lower) - images

    adv_images = (images + delta).detach()

    with torch.no_grad():
        adv_logits = model(adv_images)
        adv_preds = adv_logits.argmax(dim=1)
        success = (adv_preds != labels)

    return adv_images, success


def evaluate_robustness(model, loader, device, mean, std, attack_config):
    """
    Evaluate model robustness under FGSM and PGD attacks.

    Args:
        model: classifier in eval mode
        loader: test DataLoader
        device: torch device
        mean, std: normalization params (tuples)
        attack_config: dict with 'fgsm' and 'pgd' sub-dicts

    Returns:
        dict with clean_acc, fgsm_acc, pgd_acc
    """
    model.eval()
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    fgsm_cfg = attack_config["fgsm"]
    pgd_cfg = attack_config["pgd"]

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        total += batch_size

        # Clean accuracy
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        # Only attack correctly classified images
        correct_mask = (clean_preds == labels)
        if correct_mask.sum() == 0:
            continue

        imgs_c = images[correct_mask]
        labs_c = labels[correct_mask]

        # FGSM
        _, fgsm_succ = fgsm_attack(
            model, imgs_c, labs_c,
            epsilon=fgsm_cfg["epsilon"],
            mean=mean, std=std,
        )
        fgsm_correct += correct_mask.sum().item() - fgsm_succ.sum().item()

        # PGD
        _, pgd_succ = pgd_attack(
            model, imgs_c, labs_c,
            epsilon=pgd_cfg["epsilon"],
            alpha=pgd_cfg["alpha"],
            steps=pgd_cfg["steps"],
            mean=mean, std=std,
        )
        pgd_correct += correct_mask.sum().item() - pgd_succ.sum().item()

    clean_acc = 100.0 * clean_correct / total
    fgsm_acc = 100.0 * fgsm_correct / total
    pgd_acc = 100.0 * pgd_correct / total

    return {
        "clean_acc": clean_acc,
        "fgsm_acc": fgsm_acc,
        "pgd_acc": pgd_acc,
        "total": total,
    }
