"""Verify data pipeline by loading one batch from each split and saving an image grid.

Requirements:
1. Load one batch from train split, val split, test split.
2. Print batch shapes, dtypes, and label distributions.
3. Save an image grid of the training batch to docs/data_report/training_batch_grid.png.
"""

from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinaguard_ml.data.loaders.aptos import APTOSDataset, FUNDUS_MEAN, FUNDUS_STD


def denormalize_batch(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convert normalized [B, 3, H, W] tensor back to [B, H, W, 3] uint8 for inspection."""
    arr = tensor_batch.cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, 3
    # Reverse ImageNet/fundus normalization
    arr = arr * FUNDUS_STD + FUNDUS_MEAN
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def verify_batches(
    manifest_path: Path,
    cache_dir: Path,
    batch_size: int = 16,
    grid_out: Path = Path("docs/data_report/training_batch_grid.png"),
) -> None:
    print("\n" + "=" * 65)
    print("  VERIFYING REAL DATA PIPELINE BATCHES")
    print("=" * 65)
    print(f"  Manifest : {manifest_path}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Batch sz : {batch_size}")
    print("-" * 65)

    splits = ["train", "val", "test"]
    first_train_batch = None

    for split_name in splits:
        ds = APTOSDataset(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            split=split_name,
            augment=(split_name == "train"),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split_name == "train"))

        # Fetch first batch
        for imgs, lbls in loader:
            counts = dict(Counter(lbls.numpy().tolist()))
            print(f"[{split_name.upper():5s} BATCH]")
            print(f"  Images Tensor Shape : {imgs.shape} ({imgs.dtype})")
            print(f"  Labels Tensor Shape : {lbls.shape} ({lbls.dtype})")
            print(f"  Label Counts in Batch: {counts}")
            print(f"  Pixel Range (normed): min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")

            if split_name == "train" and first_train_batch is None:
                first_train_batch = (imgs, lbls)
            break

    # Save visual grid of first training batch
    if first_train_batch is not None:
        imgs, lbls = first_train_batch
        b = imgs.shape[0]
        cols = 4
        rows = (b + cols - 1) // cols
        h, w = imgs.shape[2], imgs.shape[3]

        denorm_imgs = denormalize_batch(imgs)

        # Scale down thumbnails if large for compact visual grid
        thumb_sz = (256, 256)
        grid_img = Image.new("RGB", (cols * thumb_sz[0], rows * thumb_sz[1]), (20, 20, 20))

        for idx in range(b):
            im = Image.fromarray(denorm_imgs[idx]).resize(thumb_sz, Image.Resampling.LANCZOS)
            r = idx // cols
            c = idx % cols
            grid_img.paste(im, (c * thumb_sz[0], r * thumb_sz[1]))

        grid_out.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(grid_out, format="PNG")
        print("-" * 65)
        print(f"[+] Saved visual inspection grid of training batch to: {grid_out.resolve()}")
    print("=" * 65 + "\n")


def main() -> None:
    manifest_path = Path("data/processed/splits_v2.csv")
    cache_dir = Path("data/processed/cache_512/aptos")
    verify_batches(manifest_path, cache_dir)


if __name__ == "__main__":
    main()

