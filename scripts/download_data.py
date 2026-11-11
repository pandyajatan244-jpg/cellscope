"""Download BloodMNIST via the official medmnist package."""

from medmnist import BloodMNIST

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = BloodMNIST(split=split, download=True)
        print(f"{split:>5}: {len(ds)} samples")
    print("Done. Dataset files saved by medmnist package.")
