"""BloodMNIST loading and dataset utilities."""

import numpy as np
from medmnist import BloodMNIST
from torch.utils.data import Dataset
from torchvision import transforms

from cellscope.config import CLASS_NAMES


def get_transforms(image_size: int = 28, augment: bool = False):
    base = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    if augment:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ] + base
    return transforms.Compose(base)


def load_split(split: str, image_size: int = 28, as_numpy: bool = False):
    """Load a BloodMNIST split. split: 'train', 'val', or 'test'."""
    ds = BloodMNIST(split=split, size=image_size, download=True)
    if as_numpy:
        return np.array(ds.imgs), ds.labels.squeeze()
    return ds


def label_to_name(label: int) -> str:
    return CLASS_NAMES[label]
