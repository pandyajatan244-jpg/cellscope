import numpy as np
from cellscope.evaluate import compute_metrics


def test_perfect_accuracy():
    y = np.array([0, 1, 2, 3])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3
