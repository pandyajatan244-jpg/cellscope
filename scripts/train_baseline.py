"""Train and evaluate classical baseline models on BloodMNIST.

Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml

Outputs saved to reports/metrics/ and reports/figures/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellscope.config import CLASS_NAMES, load_config
from cellscope.data import load_split
from cellscope.evaluate import compute_metrics
from cellscope.preprocessing import extract_hog_features, flatten_images, normalize_images


def load_data(image_size: int) -> tuple:
    print("Loading splits...")
    X_train, y_train = load_split("train", image_size=image_size, as_numpy=True)
    X_val,   y_val   = load_split("val",   image_size=image_size, as_numpy=True)
    X_test,  y_test  = load_split("test",  image_size=image_size, as_numpy=True)
    print(f"  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_flat(X_train, X_val, X_test):
    return (
        flatten_images(normalize_images(X_train)),
        flatten_images(normalize_images(X_val)),
        flatten_images(normalize_images(X_test)),
    )


def prepare_hog(X_train, X_val, X_test):
    print("Extracting HOG features (train)...")
    Xh_train = extract_hog_features(X_train)
    print("Extracting HOG features (val)...")
    Xh_val   = extract_hog_features(X_val)
    print("Extracting HOG features (test)...")
    Xh_test  = extract_hog_features(X_test)
    return Xh_train, Xh_val, Xh_test


def save_confusion_matrix(cm: list, title: str, path: Path) -> None:
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


def run_experiment(
    name: str,
    model,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training: {name}")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Fit time: {train_time:.1f}s")

    t1 = time.perf_counter()
    val_pred = model.predict(X_val)
    val_inference_ms = (time.perf_counter() - t1) / len(y_val) * 1000

    val_metrics = compute_metrics(y_val, val_pred)
    print(f"  Val accuracy         : {val_metrics['accuracy']:.4f}")
    print(f"  Val balanced accuracy: {val_metrics['balanced_accuracy']:.4f}")

    # Final test evaluation
    t2 = time.perf_counter()
    test_pred = model.predict(X_test)
    test_inference_ms = (time.perf_counter() - t2) / len(y_test) * 1000

    test_metrics = compute_metrics(y_test, test_pred)
    print(f"  Test accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")

    macro_f1 = test_metrics["report"]["macro avg"]["f1-score"]
    print(f"  Test macro F1         : {macro_f1:.4f}")

    slug = name.lower().replace(" ", "_").replace("+", "")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        f"{name} — test set confusion matrix",
        figures_dir / f"confusion_{slug}.png",
    )

    record = {
        "model": name,
        "train_time_s": round(train_time, 2),
        "val_inference_ms_per_image": round(val_inference_ms, 4),
        "test_inference_ms_per_image": round(test_inference_ms, 4),
        "val": {
            "accuracy": round(val_metrics["accuracy"], 4),
            "balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
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
    }

    out_path = metrics_dir / f"{slug}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Metrics saved -> {out_path}")

    model_path = metrics_dir / f"{slug}_model.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved -> {model_path}")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = cfg["dataset"]["image_size"]
    seed = cfg["preprocessing"]["random_seed"]
    np.random.seed(seed)

    metrics_dir = Path("reports/metrics")
    figures_dir = Path("reports/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(image_size)

    # --- Logistic regression on flat pixels ---
    Xf_train, Xf_val, Xf_test = prepare_flat(X_train, X_val, X_test)

    lr_cfg = cfg["logistic_regression"]
    lr = LogisticRegression(
        max_iter=lr_cfg["max_iter"],
        C=lr_cfg["C"],
        solver=lr_cfg["solver"],
        random_state=lr_cfg["random_state"],
    )
    lr_record = run_experiment(
        "Logistic Regression",
        lr,
        Xf_train, y_train,
        Xf_val,   y_val,
        Xf_test,  y_test,
        metrics_dir, figures_dir,
    )

    # --- HOG + SVM ---
    Xh_train, Xh_val, Xh_test = prepare_hog(X_train, X_val, X_test)

    svm_cfg = cfg["svm"]
    svm = LinearSVC(
        C=svm_cfg["C"],
        max_iter=svm_cfg["max_iter"],
        random_state=svm_cfg["random_state"],
    )
    svm_record = run_experiment(
        "HOG + SVM",
        svm,
        Xh_train, y_train,
        Xh_val,   y_val,
        Xh_test,  y_test,
        metrics_dir, figures_dir,
    )

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Model':<25} {'Bal Acc':>9} {'Macro F1':>10} {'Test Acc':>10}")
    print("-" * 55)
    for r in [lr_record, svm_record]:
        print(
            f"{r['model']:<25} "
            f"{r['test']['balanced_accuracy']:>9.4f} "
            f"{r['test']['macro_f1']:>10.4f} "
            f"{r['test']['accuracy']:>10.4f}"
        )

    summary_path = metrics_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump([lr_record, svm_record], f, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
