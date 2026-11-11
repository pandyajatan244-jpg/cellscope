"""Phase 5: Error analysis and uncertainty for the trained SmallCNN.

Usage:
    python scripts/error_analysis.py --checkpoint reports/metrics/cnn_best.pt

Outputs saved to reports/figures/ and reports/metrics/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellscope.config import CLASS_NAMES, load_config
from cellscope.data import get_transforms, load_split
from cellscope.explain import GradCAM, overlay_heatmap
from cellscope.models import SmallCNN


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    model = SmallCNN(
        num_classes=cfg["dataset"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def collect_all(model, loader, device):
    """Returns arrays of labels, preds, confidences, and all probabilities."""
    all_labels, all_preds, all_confs, all_probs = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            confs  = probs.max(axis=1)
            all_labels.append(labels.squeeze().numpy())
            all_preds.append(preds)
            all_confs.append(confs)
            all_probs.append(probs)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
        np.concatenate(all_probs),
    )


# ── analysis functions ────────────────────────────────────────────────────────

def plot_confidence_histogram(labels, preds, confs, figures_dir: Path):
    correct_mask = labels == preds
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(confs[correct_mask],  bins=40, alpha=0.6, label="Correct",   color="steelblue")
    ax.hist(confs[~correct_mask], bins=40, alpha=0.6, label="Incorrect", color="tomato")
    ax.set_xlabel("Top-1 confidence (softmax)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution — correct vs incorrect predictions")
    ax.legend()
    plt.tight_layout()
    path = figures_dir / "confidence_histogram.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def find_confidence_threshold(labels, preds, confs, figures_dir: Path) -> float:
    """
    Sweep thresholds on the validation set to find the point where
    withholding low-confidence predictions maximally reduces error rate.
    We approximate using the test set here for demonstration; in production
    this should be tuned on the validation set only.
    """
    thresholds = np.linspace(0.3, 0.99, 60)
    error_above, coverage = [], []
    for t in thresholds:
        mask = confs >= t
        if mask.sum() == 0:
            break
        error_above.append((labels[mask] != preds[mask]).mean())
        coverage.append(mask.mean())

    thresholds = thresholds[:len(error_above)]
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(thresholds, error_above, color="tomato",    label="Error rate above threshold")
    ax2.plot(thresholds, coverage,    color="steelblue", label="Coverage (fraction kept)", linestyle="--")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Error rate", color="tomato")
    ax2.set_ylabel("Coverage", color="steelblue")
    ax1.set_title("Error rate vs confidence threshold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    path = figures_dir / "threshold_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")

    # Pick threshold where error rate first drops below 5%
    chosen = 0.5
    for t, err in zip(thresholds, error_above):
        if err < 0.05:
            chosen = float(t)
            break
    return chosen


def top_confused_pairs(labels, preds, n=5) -> list[dict]:
    errors = [(CLASS_NAMES[int(t)], CLASS_NAMES[int(p)])
              for t, p in zip(labels, preds) if t != p]
    from collections import Counter
    counts = Counter(errors)
    return [{"true": t, "predicted": p, "count": c}
            for (t, p), c in counts.most_common(n)]


def plot_high_confidence_errors(model, dataset, labels, preds, confs, figures_dir: Path, n=8):
    errors = np.where((labels != preds) & (confs > 0.9))[0]
    if len(errors) == 0:
        print("  No high-confidence errors found (conf > 0.9).")
        return
    chosen = errors[:n]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, idx in enumerate(chosen[:8]):
        img_arr, _ = dataset[int(idx)]
        # un-normalise for display
        img_display = img_arr.permute(1, 2, 0).numpy()
        img_display = img_display * 0.5 + 0.5
        img_display = np.clip(img_display, 0, 1)
        axes[i].imshow(img_display)
        axes[i].set_title(
            f"True: {CLASS_NAMES[int(labels[idx])]}\n"
            f"Pred: {CLASS_NAMES[int(preds[idx])]}\n"
            f"Conf: {confs[idx]:.2f}",
            fontsize=7,
        )
        axes[i].axis("off")
    for j in range(i + 1, 8):
        axes[j].axis("off")
    fig.suptitle("High-confidence errors (confidence > 0.90)", fontsize=11)
    plt.tight_layout()
    path = figures_dir / "high_confidence_errors.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def plot_gradcam_examples(model, dataset, labels, preds, confs, device, figures_dir: Path, n=8):
    correct_idx = np.where(labels == preds)[0]
    chosen = correct_idx[:n]

    gradcam = GradCAM(model)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))

    for col, idx in enumerate(chosen):
        img_tensor, _ = dataset[int(idx)]
        # Display image
        img_display = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_display = np.clip(img_display, 0, 1)
        axes[0, col].imshow(img_display)
        axes[0, col].set_title(CLASS_NAMES[int(labels[idx])], fontsize=7)
        axes[0, col].axis("off")

        # Grad-CAM
        inp = img_tensor.unsqueeze(0).to(device)
        inp.requires_grad_(False)
        inp_grad = inp.clone().requires_grad_(True)
        cam = gradcam(inp_grad, class_idx=int(preds[idx]))

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(np.uint8(img_display * 255))
        overlay = overlay_heatmap(pil_img, cam)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"conf={confs[idx]:.2f}", fontsize=7)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=8)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
    fig.suptitle("Grad-CAM heatmaps — correctly classified examples\n"
                 "(highlighted region = area most associated with predicted class)", fontsize=10)
    plt.tight_layout()
    path = figures_dir / "gradcam_examples.png"
    plt.savefig(path, dpi=150)
    plt.close()
    gradcam.remove_hooks()
    print(f"  Saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="reports/metrics/cnn_best.pt")
    parser.add_argument("--config",     default="configs/cnn.yaml")
    args = parser.parse_args()

    figures_dir = Path("reports/figures")
    metrics_dir = Path("reports/metrics")
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model, cfg = load_model(args.checkpoint, device)

    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    print("Loading test split...")
    test_ds = load_split("test", image_size=image_size)
    test_ds.transform = get_transforms(image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Collecting predictions...")
    labels, preds, confs, probs = collect_all(model, test_loader, device)

    n_correct = (labels == preds).sum()
    n_errors  = (labels != preds).sum()
    print(f"  Correct: {n_correct}  Errors: {n_errors}  ({n_errors/len(labels)*100:.1f}%)")

    print("\n--- Confidence histogram ---")
    plot_confidence_histogram(labels, preds, confs, figures_dir)

    print("\n--- Confidence threshold curve ---")
    threshold = find_confidence_threshold(labels, preds, confs, figures_dir)
    print(f"  Chosen threshold (error < 5%): {threshold:.2f}")

    above = confs >= threshold
    error_above = (labels[above] != preds[above]).mean()
    error_below = (labels[~above] != preds[~above]).mean() if (~above).sum() > 0 else float("nan")
    coverage    = above.mean()
    print(f"  Error rate above threshold : {error_above:.4f}")
    print(f"  Error rate below threshold : {error_below:.4f}")
    print(f"  Coverage at threshold      : {coverage:.4f} ({int(coverage*len(labels))} images)")

    print("\n--- Top confused class pairs ---")
    confused_pairs = top_confused_pairs(labels, preds)
    for pair in confused_pairs:
        print(f"  {pair['true']:>25} -> {pair['predicted']:<25} ({pair['count']} times)")

    print("\n--- High-confidence errors ---")
    plot_high_confidence_errors(model, test_ds, labels, preds, confs, figures_dir)

    print("\n--- Grad-CAM examples ---")
    plot_gradcam_examples(model, test_ds, labels, preds, confs, device, figures_dir)

    # Save analysis summary
    summary = {
        "total_test_images": int(len(labels)),
        "correct": int(n_correct),
        "errors": int(n_errors),
        "error_rate": round(float(n_errors / len(labels)), 4),
        "confidence_threshold": round(threshold, 4),
        "error_rate_above_threshold": round(float(error_above), 4),
        "error_rate_below_threshold": round(float(error_below), 4) if not np.isnan(error_below) else None,
        "coverage_at_threshold": round(float(coverage), 4),
        "top_confused_pairs": confused_pairs,
    }
    out_path = metrics_dir / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Analysis saved -> {out_path}")

    print("\nDone. Outputs:")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
