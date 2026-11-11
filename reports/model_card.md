# Model Card — CellScope SmallCNN

## Model purpose

Educational blood-cell image classification on BloodMNIST. Classifies 28×28 RGB microscopy images into one of eight blood-cell categories and provides confidence scores, uncertainty warnings, and Grad-CAM visual explanations.

## Intended users

Researchers, students, and software engineers reviewing this portfolio project.

## Intended use

Learning and demonstration of biomedical image classification, reproducible ML experimentation, model evaluation, and explainability techniques.

## Out-of-scope uses

Clinical diagnosis, medical decision support, patient triage, or any application where predictions could influence a medical outcome.

## Training data

BloodMNIST (MedMNIST v2) — training split, 11,959 images across 8 classes.

- Source: https://medmnist.com/
- License: CC BY 4.0
- Paper: Yang et al., Scientific Data 2023. https://doi.org/10.1038/s41597-022-01721-8

Class distribution (training split):

| Class | Count | % |
|---|---:|---:|
| Neutrophil | 2,330 | 19.5% |
| Eosinophil | 2,181 | 18.2% |
| Immature granulocytes | 2,026 | 16.9% |
| Platelet | 1,643 | 13.7% |
| Erythroblast | 1,085 | 9.1% |
| Monocyte | 993 | 8.3% |
| Basophil | 852 | 7.1% |
| Lymphocyte | 849 | 7.1% |

Imbalance ratio: 2.7× (Neutrophil vs Lymphocyte). Moderate — balanced accuracy and macro F1 used as primary metrics.

## Evaluation data

BloodMNIST — held-out test split, 3,421 images. Evaluated once after model and hyperparameter selection on the validation split (1,712 images).

## Preprocessing

- Images loaded as uint8 RGB (28×28×3), values 0–255
- Normalised to float32, mean 0.5 / std 0.5 per channel (range approximately –1 to +1)
- Training augmentation: random horizontal flip, random rotation ±10°
- No augmentation at validation or test time
- Random seed: 42

## Model architecture

```
SmallCNN (~619,848 trainable parameters)

Input: (batch, 3, 28, 28)

Feature extractor:
  Conv2d(3 → 32, kernel=3, pad=1) → ReLU → MaxPool2d(2)    # → (32, 14, 14)
  Conv2d(32 → 64, kernel=3, pad=1) → ReLU → MaxPool2d(2)   # → (64, 7, 7)
  Conv2d(64 → 128, kernel=3, pad=1) → ReLU → AdaptiveAvgPool2d(4)  # → (128, 4, 4)

Classifier:
  Flatten                    # → 2048
  Linear(2048 → 256) → ReLU
  Dropout(0.3)
  Linear(256 → 8)

Output: logits (8,) — softmax applied at inference time
```

## Training configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Best checkpoint epoch | 28 |
| Total training time | ~865 s (CPU) |
| Random seed | 42 |

## Metrics

Test set results (evaluated once, after model selection on validation set):

| Metric | Value |
|---|---:|
| Test accuracy | 94.86% |
| Balanced accuracy | 93.49% |
| Macro F1 | 94.12% |
| Weighted F1 | 94.89% |
| Inference time | 2.17 ms / image (CPU) |

Per-class test results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Basophil | 0.894 | 0.893 | 0.894 |
| Eosinophil | 0.983 | 0.990 | 0.987 |
| Erythroblast | 0.988 | 0.987 | 0.987 |
| Immature granulocytes | 0.895 | 0.874 | 0.884 |
| Lymphocyte | 0.967 | 0.959 | 0.963 |
| Monocyte | 0.930 | 0.921 | 0.925 |
| Neutrophil | 0.975 | 0.975 | 0.975 |
| Platelet | 0.993 | 0.994 | 0.993 |

Comparison with baselines:

| Model | Balanced accuracy | Macro F1 | Test accuracy |
|---|---:|---:|---:|
| HOG + LinearSVM | 66.18% | 67.03% | 70.71% |
| Logistic regression | 79.78% | 80.22% | 82.34% |
| **SmallCNN** | **93.49%** | **94.12%** | **94.86%** |

## Uncertainty and confidence threshold

- A confidence threshold of **0.48** was derived from the test-set analysis
- Predictions below this threshold are flagged as "Review recommended"
- Error rate above threshold: 4.88% (vs 50% below)
- Coverage at threshold: 99.4% of test images accepted
- Note: in production, this threshold must be tuned on the validation set only — not the test set

Softmax confidence is not a calibrated probability of correctness. The model may be confidently wrong, particularly for Immature granulocytes.

## Known limitations

- Trained on a single public dataset under controlled imaging conditions. Likely to degrade on images from different microscopes, staining protocols, or acquisition settings.
- Softmax confidence is not calibrated — high confidence does not guarantee a correct prediction.
- Eight-class scope does not cover all blood-cell types.
- 28×28 resolution discards fine morphological detail.
- Imbalance ratio of 2.7× means rarer classes (Basophil, Lymphocyte) have slightly lower recall.
- Immature granulocytes is the hardest class: errors from Monocyte (43), Neutrophil (21), and Basophil (21) account for the majority of mistakes.

## Ethical and safety considerations

This model must not be used for any clinical purpose. Predictions are intended solely for educational and research demonstration. No validation against clinical ground truth has been performed.

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Train CNN (reproduces cnn_best.pt with seed 42)
python scripts/train_cnn.py --config configs/cnn.yaml

# Run error analysis
python scripts/error_analysis.py --checkpoint reports/metrics/cnn_best.pt
```

Fixed random seed: 42 (Python, NumPy, PyTorch).

## Citations

```
Yang, J. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical
image classification. Scientific Data, 10, 2023.
https://doi.org/10.1038/s41597-022-01721-8
```
