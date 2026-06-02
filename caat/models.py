"""
models.py — Model architectures for CIFAR-10 and GTSRB.

ResNet18_CIFAR: Adapted ResNet18 for 32×32 inputs (3×3 conv1, no maxpool).
MobileNetV2_Small: MobileNetV2 adapted for 32×32 / small images.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18_CIFAR(nn.Module):
    """
    ResNet18 adapted for 32×32 CIFAR-10 / GTSRB images.

    Changes from standard ResNet18:
      - conv1: 3×3 kernel (not 7×7), stride=1, padding=1
      - maxpool: removed (Identity)
      - fc: output dim matches num_classes
    """
    def __init__(self, num_classes=10):
        super().__init__()
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)


class MobileNetV2_Small(nn.Module):
    """
    MobileNetV2 adapted for 32×32 images.

    Changes:
      - First conv stride: 1 (not 2)
      - Classifier: adjusted for num_classes
    """
    def __init__(self, num_classes=43):
        super().__init__()
        m = models.mobilenet_v2(weights=None)
        # Adjust first conv for small images: stride 1 instead of 2
        m.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)


def load_cifar_resnet18(checkpoint_path, device, num_classes=10):
    """Load pretrained CIFAR-10 ResNet18 from checkpoint."""
    model = ResNet18_CIFAR(num_classes=num_classes).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both direct state_dict and wrapped formats
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    # If keys don't have 'model.' prefix, add it
    # (old training script saved the inner resnet directly)
    sample_key = next(iter(sd.keys()))
    if not sample_key.startswith("model."):
        sd = {f"model.{k}": v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_imagenet_model(name, device):
    """
    Load a pretrained ImageNet model from torchvision.

    Args:
        name: one of 'resnet18', 'resnet34', 'resnet50', 'vgg16',
              'densenet121', 'mobilenet_v2', 'efficientnet_b0',
              'vit_b_16', 'vit_b_32'
        device: torch device

    Returns:
        model in eval mode
    """
    weight_map = {
        "resnet18":       (models.resnet18,       models.ResNet18_Weights.DEFAULT),
        "resnet34":       (models.resnet34,       models.ResNet34_Weights.DEFAULT),
        "resnet50":       (models.resnet50,       models.ResNet50_Weights.DEFAULT),
        "vgg16":          (models.vgg16,          models.VGG16_Weights.DEFAULT),
        "densenet121":    (models.densenet121,    models.DenseNet121_Weights.DEFAULT),
        "mobilenet_v2":   (models.mobilenet_v2,   models.MobileNet_V2_Weights.DEFAULT),
        "efficientnet_b0":(models.efficientnet_b0,models.EfficientNet_B0_Weights.DEFAULT),
        "vit_b_16":       (models.vit_b_16,       models.ViT_B_16_Weights.DEFAULT),
        "vit_b_32":       (models.vit_b_32,       models.ViT_B_32_Weights.DEFAULT),
    }
    if name not in weight_map:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(weight_map.keys())}")

    constructor, weights = weight_map[name]
    model = constructor(weights=weights).to(device)
    model.eval()
    return model


def get_imagenet_transform():
    """Standard ImageNet preprocessing: resize 256 → center crop 224 → normalize."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_cifar10_transforms(train=True):
    """CIFAR-10 train/test transforms."""
    import torchvision.transforms as T
    from .config import CIFAR10_MEAN, CIFAR10_STD

    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_gtsrb_transforms(train=True):
    """GTSRB train/test transforms."""
    import torchvision.transforms as T
    from .config import GTSRB_MEAN, GTSRB_STD

    if train:
        return T.Compose([
            T.Resize((32, 32)),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(GTSRB_MEAN, GTSRB_STD),
        ])
    else:
        return T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(GTSRB_MEAN, GTSRB_STD),
        ])
