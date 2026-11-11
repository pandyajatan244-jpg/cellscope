"""CellScope — Streamlit review interface."""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellscope.config import CLASS_NAMES
from cellscope.data import get_transforms
from cellscope.explain import GradCAM, overlay_heatmap
from cellscope.models import SmallCNN
from cellscope.predict import predict_image

# ── constants ─────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent
METRICS_DIR  = REPO_ROOT / "reports" / "metrics"
FIGURES_DIR  = REPO_ROOT / "reports" / "figures"
CHECKPOINT   = METRICS_DIR / "cnn_best.pt"
THRESHOLD    = 0.4754  # from error_analysis.json


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    cfg    = ckpt["config"]
    model  = SmallCNN(
        num_classes=cfg["dataset"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, cfg


@st.cache_data
def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_baseline_summary() -> list:
    return load_json(str(METRICS_DIR / "baseline_summary.json"))


@st.cache_data
def load_cnn_metrics() -> dict:
    return load_json(str(METRICS_DIR / "cnn_metrics.json"))


@st.cache_data
def load_error_analysis() -> dict:
    return load_json(str(METRICS_DIR / "error_analysis.json"))


# ── page helpers ──────────────────────────────────────────────────────────────

def figure_if_exists(path: Path):
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"Figure not found: {path.name}")


def prob_bar_chart(probs: dict):
    names = list(probs.keys())
    vals  = [probs[n] * 100 for n in names]
    colors = ["#1f77b4" if v != max(vals) else "#d62728" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Predicted probability (%)",
        xaxis_range=[0, 105],
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def confusion_matrix_chart(cm: list, title: str):
    cm_arr = np.array(cm)
    fig = px.imshow(
        cm_arr,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=CLASS_NAMES, y=CLASS_NAMES,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ── pages ─────────────────────────────────────────────────────────────────────

def page_home():
    st.title("CellScope")
    st.caption("Educational biomedical image classification — not for clinical use.")

    st.error(
        "**Disclaimer:** CellScope is an educational research project. "
        "Predictions must not be used for clinical diagnosis or any medical decision."
    )

    st.markdown("""
### What is this?

CellScope classifies microscopy images of blood cells into one of eight categories
and shows the model's confidence, a probability breakdown, and a visual explanation
of which image region drove the prediction.

It compares three models — a logistic-regression baseline, a HOG + linear SVM baseline,
and a small convolutional neural network — using class-aware metrics and error analysis.

### Research question

> *How reliably can lightweight image-classification models distinguish blood-cell categories
> from microscopy images, and how can a user understand when a prediction should be treated cautiously?*

### Dataset

**BloodMNIST** (MedMNIST v2) — 17,092 RGB microscopy images across 8 blood-cell classes.

| | |
|---|---|
| Source | https://medmnist.com/ |
| License | CC BY 4.0 |
| Paper | Yang et al., *Scientific Data* 2023. https://doi.org/10.1038/s41597-022-01721-8 |

### Classes
""")

    cols = st.columns(4)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 4].markdown(f"- {name}")

    st.markdown("""
### Intended use
Research and educational purposes only. This project was built to demonstrate skills
in biomedical computer vision, reproducible ML, model evaluation, and explainability.

### Limitations
- Trained on a single public dataset under controlled imaging conditions.
- May not generalise to images from different microscopes or staining protocols.
- Softmax confidence is not a calibrated probability of correctness.
- Eight-class scope does not cover all blood-cell types.
""")


def page_predict():
    st.title("Single-image prediction")
    st.caption("Upload a blood-cell microscopy image to get a prediction.")

    model, device, cfg = load_model()
    transform = get_transforms(cfg["dataset"]["image_size"], augment=False)

    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"]
    )

    st.divider()
    if uploaded is None:
        st.info("Upload an image above to see a prediction.")

        st.markdown("**No image to hand?** Try saving one of the example images from the Dataset Explorer page.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    w, h = image.size
    if w < 10 or h < 10:
        st.error("Image is too small. Please upload a valid microscopy image.")
        return

    col_img, col_result = st.columns([1, 2])

    with col_img:
        st.subheader("Input image")
        st.image(image, width=200)
        st.caption(f"{w} x {h} px  |  RGB")

    result = predict_image(model, image, transform, device, threshold=THRESHOLD)

    with col_result:
        st.subheader("Prediction")

        if result["review_recommended"]:
            st.warning(
                f"**Review recommended** — confidence {result['top1_confidence']*100:.1f}% "
                f"is below the {THRESHOLD*100:.0f}% threshold. "
                "Treat this prediction with caution."
            )
        else:
            st.success(f"**{result['predicted_class']}**  —  confidence {result['top1_confidence']*100:.1f}%")

        st.caption(f"Inference time: {result['inference_time_ms']} ms")

        st.subheader("All class probabilities")
        prob_bar_chart(result["all_probabilities"])

        st.subheader("Top-3 predictions")
        for i, p in enumerate(result["top_predictions"], 1):
            st.markdown(f"{i}. **{p['class']}** — {p['probability']*100:.1f}%")

    # Grad-CAM
    st.divider()
    st.subheader("Grad-CAM explanation")
    st.caption(
        "Highlighted regions are most associated with the predicted class. "
        "This is an indication only — not proof of biological significance."
    )

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        inp = img_tensor.clone().requires_grad_(True)
        gradcam = GradCAM(model)
        pred_idx = CLASS_NAMES.index(result["predicted_class"])
        cam = gradcam(inp, class_idx=pred_idx)
        gradcam.remove_hooks()
        overlay = overlay_heatmap(image, cam)
        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.image(image.resize((128, 128), Image.NEAREST), caption="Original", width=200)
        with col_cam:
            st.image(overlay, caption="Grad-CAM overlay", width=200)
    except Exception as e:
        st.warning(f"Grad-CAM could not be generated: {e}")


def page_dataset():
    st.title("Dataset explorer")

    st.subheader("Class distribution")
    figure_if_exists(FIGURES_DIR / "class_distribution.png")

    st.subheader("Example images — one row per class")
    figure_if_exists(FIGURES_DIR / "example_grid.png")

    st.subheader("Pixel intensity distributions")
    figure_if_exists(FIGURES_DIR / "pixel_distribution.png")

    st.markdown("""
**Dataset notes:**
- 17,092 images across 8 classes — moderate imbalance (2.7× ratio, Neutrophil vs Lymphocyte)
- All images are 28 × 28 RGB, uint8, values 0–255
- Standard train / val / test splits: 11,959 / 1,712 / 3,421
- Imbalance ratio of 2.7× is moderate — balanced accuracy and macro F1 are the primary metrics
""")


def page_evaluation():
    st.title("Model evaluation")

    cnn = load_cnn_metrics()
    ea  = load_error_analysis()

    st.subheader("CNN — test set results")
    m = cnn["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy",      f"{m['accuracy']*100:.2f}%")
    c2.metric("Balanced accuracy",  f"{m['balanced_accuracy']*100:.2f}%")
    c3.metric("Macro F1",           f"{m['macro_f1']*100:.2f}%")
    c4.metric("Weighted F1",        f"{m['weighted_f1']*100:.2f}%")

    st.subheader("Confusion matrix")
    confusion_matrix_chart(cnn["confusion_matrix"], "SmallCNN — test set")

    st.subheader("Per-class metrics")
    import pandas as pd
    rows = [
        {"Class": cls, **{k: f"{v:.3f}" for k, v in vals.items()}}
        for cls, vals in m["per_class"].items()
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    st.subheader("Training curve")
    figure_if_exists(FIGURES_DIR / "cnn_training_curve.png")

    st.subheader("Confidence analysis")
    ea_cols = st.columns(3)
    ea_cols[0].metric("Error rate (all)",          f"{ea['error_rate']*100:.1f}%")
    ea_cols[1].metric("Error rate above threshold", f"{ea['error_rate_above_threshold']*100:.1f}%")
    ea_cols[2].metric("Coverage at threshold",      f"{ea['coverage_at_threshold']*100:.1f}%")

    st.caption(
        f"Confidence threshold: {ea['confidence_threshold']:.2f}  |  "
        "Predictions below this threshold are flagged as 'Review recommended'."
    )

    col_hist, col_thresh = st.columns(2)
    with col_hist:
        figure_if_exists(FIGURES_DIR / "confidence_histogram.png")
    with col_thresh:
        figure_if_exists(FIGURES_DIR / "threshold_curve.png")

    st.subheader("High-confidence errors")
    st.caption("Images the model predicted incorrectly with confidence above 90%.")
    figure_if_exists(FIGURES_DIR / "high_confidence_errors.png")

    st.subheader("Grad-CAM examples")
    st.caption(
        "Correctly classified examples with heatmap overlay. "
        "Red = region most associated with the predicted class."
    )
    figure_if_exists(FIGURES_DIR / "gradcam_examples.png")

    st.subheader("Most confused class pairs")
    for pair in ea["top_confused_pairs"]:
        st.markdown(f"- **{pair['true']}** predicted as **{pair['predicted']}** — {pair['count']} times")

    st.info(
        "**Pattern:** Immature granulocytes absorb most errors. "
        "This is expected — they share morphological features with Monocyte, "
        "Neutrophil, Basophil, and Lymphocyte at 28×28 resolution."
    )


def page_comparison():
    st.title("Model comparison")

    baselines = load_baseline_summary()
    cnn       = load_cnn_metrics()

    import pandas as pd

    rows = []
    for b in baselines:
        rows.append({
            "Model":             b["model"],
            "Input":             "Flat pixels" if "Logistic" in b["model"] else "HOG features",
            "Test accuracy":     f"{b['test']['accuracy']*100:.2f}%",
            "Balanced accuracy": f"{b['test']['balanced_accuracy']*100:.2f}%",
            "Macro F1":          f"{b['test']['macro_f1']*100:.2f}%",
            "Train time (s)":    b["train_time_s"],
            "Inference (ms/img)": b["test_inference_ms_per_image"],
        })
    rows.append({
        "Model":             "SmallCNN",
        "Input":             "28x28 RGB",
        "Test accuracy":     f"{cnn['test']['accuracy']*100:.2f}%",
        "Balanced accuracy": f"{cnn['test']['balanced_accuracy']*100:.2f}%",
        "Macro F1":          f"{cnn['test']['macro_f1']*100:.2f}%",
        "Train time (s)":    cnn["total_train_time_s"],
        "Inference (ms/img)": cnn["test_inference_ms_per_image"],
    })

    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df, use_container_width=True)

    st.markdown("""
**Key observations:**
- The CNN improves on logistic regression by ~13 points in balanced accuracy and ~14 points in macro F1
- Logistic regression on flat pixels outperforms HOG+LinearSVM — HOG was designed for larger images; at 28×28 the spatial features lose resolution
- The CNN is slower to train (CPU ~14 min) but inference is only 2 ms per image
- All three models were evaluated on the same held-out test set, selected once after model development
""")

    st.subheader("Confusion matrices side by side")
    col1, col2 = st.columns(2)
    with col1:
        figure_if_exists(FIGURES_DIR / "confusion_logistic_regression.png")
        st.caption("Logistic Regression")
    with col2:
        figure_if_exists(FIGURES_DIR / "confusion_cnn.png")
        st.caption("SmallCNN")

    st.subheader("CNN — parameters")
    st.markdown(f"- **Trainable parameters:** {cnn['num_parameters']:,}")
    st.markdown(f"- **Best epoch:** {cnn['best_epoch']} / 30 (early stopping patience = 5)")
    st.markdown(f"- **Architecture:** 3 conv blocks (32→64→128 filters), dropout 0.3, fully connected head")


# ── app entry point ───────────────────────────────────────────────────────────

st.set_page_config(page_title="CellScope", layout="wide", page_icon="🔬")

PAGES = {
    "Home":             page_home,
    "Predict":          page_predict,
    "Dataset explorer": page_dataset,
    "Evaluation":       page_evaluation,
    "Model comparison": page_comparison,
}

with st.sidebar:
    st.title("CellScope")
    st.caption("Blood-cell image classifier")
    st.divider()
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Not for clinical use.")

PAGES[selection]()
