"""Load and validate YAML configuration files."""

from pathlib import Path
import yaml


CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature granulocytes",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]

NUM_CLASSES = len(CLASS_NAMES)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
