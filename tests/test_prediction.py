import torch
from cellscope.models import SmallCNN
from cellscope.config import NUM_CLASSES


def test_model_output_shape():
    model = SmallCNN(num_classes=NUM_CLASSES)
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_probabilities_sum_to_one():
    model = SmallCNN(num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(1, 3, 28, 28)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
