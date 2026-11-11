"""Grad-CAM explainability for SmallCNN."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Compute Grad-CAM heatmaps for SmallCNN's last conv layer."""

    def __init__(self, model):
        self.model = model
        self._activations = None
        self._gradients = None
        # Hook onto the last conv layer (index 6 in model.features: Conv->ReLU->Pool x3)
        target_layer = model.features[-1]  # AdaptiveAvgPool — use layer before it
        # Walk back to find the last ReLU (index 7 is AdaptiveAvgPool, 6 is ReLU, 5 is Conv)
        target_layer = model.features[7]   # AdaptiveAvgPool2d — we want the conv before
        # features: 0=Conv, 1=ReLU, 2=MaxPool, 3=Conv, 4=ReLU, 5=MaxPool, 6=Conv, 7=ReLU, 8=AdaptiveAvgPool
        target_layer = model.features[6]   # last Conv2d
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        tensor: (1, C, H, W) — already on the right device
        Returns a (H, W) heatmap normalised to [0, 1].
        """
        self.model.eval()
        logits = self.model(tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H, W)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a Grad-CAM heatmap on a PIL image. Returns a PIL image."""
    import matplotlib

    img_rgb = image.convert("RGB").resize((128, 128), Image.NEAREST)
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize((128, 128), Image.BILINEAR)
    cam_arr = np.array(cam_resized) / 255.0

    colormap = matplotlib.colormaps["jet"]
    heatmap_rgb = np.uint8(colormap(cam_arr)[:, :, :3] * 255)
    blended = np.uint8((1 - alpha) * np.array(img_rgb) + alpha * heatmap_rgb)
    return Image.fromarray(blended)
