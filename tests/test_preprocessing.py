import numpy as np
from cellscope.preprocessing import normalize_images, flatten_images


def test_normalize_range():
    images = np.random.randint(0, 256, (10, 28, 28, 3), dtype=np.uint8)
    out = normalize_images(images)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_flatten_shape():
    images = np.zeros((5, 28, 28, 3))
    out = flatten_images(images)
    assert out.shape == (5, 28 * 28 * 3)
