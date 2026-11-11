# Data

## BloodMNIST

- **Dataset:** BloodMNIST (part of MedMNIST v2)
- **Official URL:** https://medmnist.com/
- **Repository:** https://github.com/MedMNIST/MedMNIST
- **Version:** MedMNIST v2
- **Paper:** Yang et al., Scientific Data 2023, https://doi.org/10.1038/s41597-022-01721-8
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) — see https://medmnist.com/#license for dataset-specific details
- **Date accessed:** 2026-11-09

### Contents

17,092 RGB microscopy images (28 × 28 pixels) across 8 blood-cell classes:
Basophil, Eosinophil, Erythroblast, Immature granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet.

Standard splits: train / validation / test.

### Download method

Downloaded via the official `medmnist` Python package:

```python
import medmnist
from medmnist import BloodMNIST
```

### Storage

Raw dataset files are excluded from version control (see `.gitignore`).
Download using `scripts/download_data.py`.

### Intended use

Research and educational purposes only. Not intended for clinical diagnosis or treatment decisions.
