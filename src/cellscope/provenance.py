"""Data provenance and dataset metadata."""

import json
from datetime import date
from pathlib import Path


PROVENANCE = {
    "dataset": "BloodMNIST",
    "source_url": "https://medmnist.com/",
    "repository": "https://github.com/MedMNIST/MedMNIST",
    "paper_doi": "https://doi.org/10.1038/s41597-022-01721-8",
    "license": "CC BY 4.0",
    "download_method": "medmnist Python package",
    "intended_use": "Research and educational purposes only. Not for clinical diagnosis.",
}


def save_provenance(output_path: str | Path, extra: dict | None = None) -> None:
    record = {**PROVENANCE, "date_recorded": str(date.today())}
    if extra:
        record.update(extra)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
