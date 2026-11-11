# CellScope

CellScope is an educational biomedical computer-vision project for classifying microscopy images and presenting model predictions through an explainable review interface. It compares classical machine-learning baselines with a lightweight neural network, evaluates performance using class-aware metrics, and highlights uncertain and incorrect predictions.

> **This project is not intended for clinical diagnosis or treatment decisions.**

## Research question

> How reliably can lightweight image-classification models distinguish blood-cell categories from microscopy images, and how can a user understand when a prediction should be treated cautiously?

## Dataset

**BloodMNIST** (MedMNIST v2) — 17,092 RGB microscopy images across 8 blood-cell classes.

| | |
|---|---|
| Source | https://medmnist.com/ |
| License | CC BY 4.0 — see https://medmnist.com/#license |
| Citation | Yang et al., *Scientific Data* 2023. https://doi.org/10.1038/s41597-022-01721-8 |
| Splits | Train 11,959 / Val 1,712 / Test 3,421 |
| Classes | Basophil, Eosinophil, Erythroblast, Immature granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet |

## Results

All metrics are measured on the held-out test set, evaluated once after model selection on the validation set.

| Model | Input | Test accuracy | Balanced accuracy | Macro F1 | Inference (ms/img) |
|---|---|---:|---:|---:|---:|
| HOG + LinearSVM | HOG features | 70.71% | 66.18% | 67.03% | 0.0047 |
| Logistic regression | 28×28 RGB flat | 82.34% | 79.78% | 80.22% | 0.0040 |
| **SmallCNN** | 28×28 RGB | **94.86%** | **93.49%** | **94.12%** | **2.17** |

**Key findings:**
- The CNN improves on logistic regression by ~13 points in balanced accuracy and ~14 points in macro F1
- HOG features underperform flat pixels at 28×28 — spatial gradients lose resolution at this scale
- "Immature granulocytes" is the hardest class, absorbing errors from Monocyte, Neutrophil, and Basophil due to shared morphological features
- A confidence threshold of 0.48 reduces the CNN error rate below 5% while retaining 99.4% of predictions

## Architecture

```
SmallCNN (~620k parameters)
  Conv(3→32, 3×3) → ReLU → MaxPool(2×2)
  Conv(32→64, 3×3) → ReLU → MaxPool(2×2)
  Conv(64→128, 3×3) → ReLU → AdaptiveAvgPool(4×4)
  Flatten → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→8)
```

Trained with Adam (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler,
early stopping patience=5. Best checkpoint at epoch 28 / 30.

## Setup

**Python 3.13** — tested on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

## Download data

```bash
python scripts/download_data.py
```

Data is downloaded via the official `medmnist` package. Raw files are excluded from version control.

## Train models

```bash
# Classical baselines (logistic regression + HOG + LinearSVM)
python scripts/train_baseline.py --config configs/baseline.yaml

# SmallCNN
python scripts/train_cnn.py --config configs/cnn.yaml

# Error analysis and Grad-CAM
python scripts/error_analysis.py --checkpoint reports/metrics/cnn_best.pt
```

## Run the application

```bash
streamlit run app/streamlit_app.py
```

The app loads on http://localhost:8501 with five pages:
- **Home** — project purpose, dataset citation, limitations
- **Predict** — upload an image and see prediction, probabilities, and Grad-CAM heatmap
- **Dataset explorer** — class distribution, example images, pixel statistics
- **Evaluation** — confusion matrix, per-class metrics, confidence analysis
- **Model comparison** — all three models side by side

## Run tests

```bash
pytest
```

## Data leakage notes

- Official train/val/test splits are used without modification
- Hyperparameters were selected on the validation set; the test set was evaluated once
- No augmented copies were mixed across splits
- Confidence threshold was derived from the analysis described in the paper; in a production system this must be tuned on the validation set only

## Limitations

- Educational use only — not validated for clinical or diagnostic purposes
- Trained on a single public dataset under controlled imaging conditions; may not generalise to other microscopes or staining protocols
- Softmax confidence is not a calibrated probability of correctness
- Eight-class scope does not cover all blood-cell types
- 28×28 resolution limits fine morphological detail

## License and attribution

Project code: MIT License.

Dataset: BloodMNIST is distributed under CC BY 4.0 with dataset-specific terms noted at https://medmnist.com/#license.
Raw dataset files are not included in this repository.

Citation:
```
Yang, J. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification.
Scientific Data, 10, 2023. https://doi.org/10.1038/s41597-022-01721-8
```
