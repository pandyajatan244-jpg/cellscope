"""Image preprocessing and feature extraction utilities."""

import numpy as np
from skimage.feature import hog


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1]."""
    return images.astype(np.float32) / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten (N, H, W, C) to (N, H*W*C)."""
    return images.reshape(len(images), -1)


def extract_hog_features(images: np.ndarray, **kwargs) -> np.ndarray:
    """Extract HOG features from a batch of images."""
    features = [
        hog(img, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), channel_axis=-1, **kwargs)
        for img in images
    ]
    return np.array(features)
