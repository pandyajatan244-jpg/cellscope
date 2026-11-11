"""Inference utilities."""

import time
import numpy as np
import torch
from PIL import Image

from cellscope.config import CLASS_NAMES


def predict_image(model, image: Image.Image, transform, device: str, threshold: float = 0.5) -> dict:
    model.eval()
    tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    inference_ms = (time.perf_counter() - start) * 1000

    top3_idx = np.argsort(probs)[::-1][:3]

    return {
        "predicted_class": CLASS_NAMES[int(np.argmax(probs))],
        "top_predictions": [
            {"class": CLASS_NAMES[i], "probability": float(probs[i])} for i in top3_idx
        ],
        "all_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "top1_confidence": float(probs.max()),
        "review_recommended": float(probs.max()) < threshold,
        "inference_time_ms": round(inference_ms, 2),
    }
