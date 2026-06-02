"""
dataset.py — Adversarial/clean image pair dataset for PCSD training.

Generates adversarial examples on-the-fly or loads pre-generated pairs.
Each sample: (x_adv, x_clean, label, vulnerability_profile)
"""

import torch
from torch.utils.data import Dataset


class AdversarialPairDataset(Dataset):
    """
    Dataset of (adversarial, clean) image pairs for denoiser training.

    During PCSD training, we need pairs of:
      - x_clean: original clean image (in [0,1] pixel space)
      - x_adv: adversarial version (in [0,1] pixel space)
      - y: true label
      - v: vulnerability profile (same for all images of this model)

    Args:
        clean_images: (N, 3, H, W) tensor in [0, 1]
        adv_images: (N, 3, H, W) tensor in [0, 1]
        labels: (N,) tensor of labels
        vulnerability_profile: (n_bins,) numpy array
    """
    def __init__(self, clean_images, adv_images, labels, vulnerability_profile):
        self.clean = clean_images
        self.adv = adv_images
        self.labels = labels
        self.v = torch.from_numpy(vulnerability_profile).float()

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (
            self.adv[idx],
            self.clean[idx],
            self.labels[idx],
            self.v,
        )


def generate_adversarial_pairs(model, loader, device, epsilon, alpha, steps,
                                mean, std, max_images=5000):
    """
    Generate adversarial/clean image pairs for PCSD training.

    Uses PGD to create adversarial examples. Stores images in [0,1] space
    (denormalized) since the denoiser operates in pixel space.

    Args:
        model: target classifier
        loader: DataLoader with normalized images
        device: torch device
        epsilon, alpha, steps: PGD parameters
        mean, std: normalization params
        max_images: limit

    Returns:
        clean_images: (N, 3, H, W) in [0, 1]
        adv_images: (N, 3, H, W) in [0, 1]
        labels: (N,)
    """
    from caat.attacks import pgd_attack

    model.eval()
    all_clean = []
    all_adv = []
    all_labels = []
    count = 0
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    for images, labels in loader:
        if count >= max_images:
            break

        images, labels = images.to(device), labels.to(device)

        # Only use correctly classified images
        with torch.no_grad():
            preds = model(images).argmax(1)
            correct = (preds == labels)

        if correct.sum() == 0:
            continue

        imgs_c = images[correct]
        labs_c = labels[correct]

        # Generate PGD adversarial examples
        adv_imgs, _ = pgd_attack(
            model, imgs_c, labs_c,
            epsilon=epsilon, alpha=alpha, steps=steps,
            mean=mean, std=std,
        )

        # Denormalize to [0, 1] pixel space for denoiser
        clean_pixel = (imgs_c * std_t + mean_t).clamp(0, 1)
        adv_pixel = (adv_imgs * std_t + mean_t).clamp(0, 1)

        all_clean.append(clean_pixel.cpu())
        all_adv.append(adv_pixel.cpu())
        all_labels.append(labs_c.cpu())
        count += len(imgs_c)

    return (
        torch.cat(all_clean)[:max_images],
        torch.cat(all_adv)[:max_images],
        torch.cat(all_labels)[:max_images],
    )
