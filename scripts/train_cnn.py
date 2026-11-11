"""Train the SmallCNN on BloodMNIST.

Usage:
    python scripts/train_cnn.py --config configs/cnn.yaml

Outputs saved to reports/metrics/ and reports/figures/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellscope.config import CLASS_NAMES, load_config
from cellscope.data import get_transforms, load_split
from cellscope.evaluate import compute_metrics
from cellscope.models import SmallCNN
from cellscope.train import evaluate_epoch, train_epoch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(image_size: int, batch_size: int) -> tuple:
    train_ds = load_split("train", image_size=image_size)
    train_ds.transform = get_transforms(image_size, augment=True)

    val_ds = load_split("val", image_size=image_size)
    val_ds.transform = get_transforms(image_size, augment=False)

    test_ds = load_split("test", image_size=image_size)
    test_ds.transform = get_transforms(image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    return train_loader, val_loader, test_loader


def collect_predictions(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.squeeze().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def save_training_curve(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Val accuracy", color="steelblue")
    if history.get("best_epoch"):
        ax2.axvline(history["best_epoch"], color="red", linestyle="--", label=f"Best (ep {history['best_epoch']})")
    ax2.set_title("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curve saved -> {path}")


def save_confusion_matrix(cm: list, title: str, path: Path) -> None:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    p_cfg = cfg["paths"]
    image_size  = cfg["dataset"]["image_size"]
    num_classes = cfg["dataset"]["num_classes"]

    set_seed(t_cfg["random_seed"])

    metrics_dir = Path(p_cfg["checkpoint_dir"])
    figures_dir = Path(p_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = build_loaders(image_size, t_cfg["batch_size"])

    model = SmallCNN(num_classes=num_classes, dropout=m_cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = metrics_dir / "cnn_best.pt"

    print(f"\nTraining for up to {t_cfg['num_epochs']} epochs "
          f"(early stopping patience={t_cfg['early_stopping_patience']})...")

    total_train_start = time.perf_counter()

    for epoch in range(1, t_cfg["num_epochs"] + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        improved = v_loss < best_val_loss
        if improved:
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "val_acc": v_acc,
                "class_names": CLASS_NAMES,
                "config": cfg,
            }, checkpoint_path)
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"  Epoch {epoch:>3}/{t_cfg['num_epochs']}  "
              f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}{marker}")

        if patience_counter >= t_cfg["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break

    total_train_time = time.perf_counter() - total_train_start
    history["best_epoch"] = best_epoch
    print(f"\nBest checkpoint: epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    print(f"Total training time: {total_train_time:.1f}s")

    # --- Test evaluation using best checkpoint ---
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    t_inf_start = time.perf_counter()
    y_true, y_pred = collect_predictions(model, test_loader, device)
    test_inference_ms = (time.perf_counter() - t_inf_start) / len(y_true) * 1000

    test_metrics = compute_metrics(y_true, y_pred)
    macro_f1 = test_metrics["report"]["macro avg"]["f1-score"]

    print(f"  Test accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Test macro F1         : {macro_f1:.4f}")
    print(f"  Inference time        : {test_inference_ms:.3f} ms/image")

    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        "SmallCNN - test set confusion matrix",
        figures_dir / "confusion_cnn.png",
    )
    save_training_curve(history, figures_dir / "cnn_training_curve.png")

    record = {
        "model": "SmallCNN",
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "total_train_time_s": round(total_train_time, 1),
        "test_inference_ms_per_image": round(test_inference_ms, 4),
        "val": {
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(ckpt["val_acc"], 4),
        },
        "test": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "balanced_accuracy": round(test_metrics["balanced_accuracy"], 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(test_metrics["report"]["weighted avg"]["f1-score"], 4),
            "per_class": {
                cls: {
                    "precision": round(test_metrics["report"][cls]["precision"], 4),
                    "recall": round(test_metrics["report"][cls]["recall"], 4),
                    "f1": round(test_metrics["report"][cls]["f1-score"], 4),
                }
                for cls in CLASS_NAMES
            },
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "training_history": history,
    }

    out_path = metrics_dir / "cnn_metrics.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Metrics saved -> {out_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {out_path}")


if __name__ == "__main__":
    main()
