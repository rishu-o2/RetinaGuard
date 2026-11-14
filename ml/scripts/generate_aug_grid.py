"""Generate a 4x4 grid of 16 camera-augmented training images and benchmark a 1-epoch subset run."""

import csv
import shutil
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset

# Add ml/src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinaguard_ml.data.augmentations import FundusCameraAugmentor
from retinaguard_ml.data.loaders.aptos import APTOSDataset
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss

def generate_grid():
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = repo_root / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "aug_grid.png"

    manifest_path = repo_root / "data" / "processed" / "splits_v2.csv"
    cache_dir = repo_root / "data" / "processed" / "cache_512" / "aptos"

    # Pick 16 training records (diverse grades 0-4)
    train_records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_records.append(row)

    # Sort/select to cover multiple grades
    by_grade = {g: [] for g in range(5)}
    for r in train_records:
        by_grade[int(r["diagnosis"])].append(r)

    selected = []
    # Pick 3 of each grade (15) + 1 more (16 total)
    for g in range(5):
        selected.extend(by_grade[g][:3])
    selected.append(by_grade[4][3] if len(by_grade[4]) > 3 else by_grade[2][3])
    selected = selected[:16]

    augmentor = FundusCameraAugmentor(seed=42)
    grid_tiles = []
    target_tile_size = 256

    for r in selected:
        id_code = r["id_code"]
        grade = r["diagnosis"]
        img_p = cache_dir / f"{id_code}.png"
        if not img_p.exists():
            img_p = cache_dir / f"{id_code}.jpg"

        with Image.open(img_p) as pil_img:
            arr = np.array(pil_img.convert("RGB"))

        arr = cv2.resize(arr, (target_tile_size, target_tile_size), interpolation=cv2.INTER_AREA)
        # Apply camera augmentation
        aug_arr = augmentor(arr)

        # Draw overlay info
        tile = Image.fromarray(aug_arr)
        draw = ImageDraw.Draw(tile)
        draw.rectangle([(4, 4), (160, 26)], fill=(0, 0, 0, 180))
        draw.text((8, 8), f"G{grade} | {id_code[:8]}", fill=(255, 255, 255))
        grid_tiles.append(np.array(tile))

    # Assemble 4x4 grid
    rows = []
    for i in range(4):
        row = np.hstack(grid_tiles[i * 4 : (i + 1) * 4])
        rows.append(row)
    grid = np.vstack(rows)

    grid_img = Image.fromarray(grid)
    grid_img.save(out_path)
    print(f"[Grid] Saved 16-image augmentation grid to {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    # Copy to artifact dir if exists
    artifact_dir = Path(r"C:\Users\hp\.gemini\antigravity\brain\0fffc3b9-96ae-4fec-8b9f-26d6505e3a54")
    if artifact_dir.exists():
        shutil.copy2(out_path, artifact_dir / "aug_grid.png")
        print(f"[Grid] Copied grid to artifact dir: {artifact_dir / 'aug_grid.png'}")


def benchmark_subset_run():
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "processed" / "splits_v2.csv"
    cache_dir = repo_root / "data" / "processed" / "cache_512" / "aptos"

    train_ds = APTOSDataset(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        split="train",
        augment=True,
        image_size=(256, 256),
    )

    # 200-image subset
    subset_indices = list(range(min(200, len(train_ds))))
    subset_ds = Subset(train_ds, subset_indices)
    batch_size = 16
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cpu")
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False, num_classes=5).to(device)
    model.train()

    criterion = JointCategoricalOrdinalLoss(
        class_weights=torch.ones(5).to(device),
        ordinal_weight=0.50,
        num_classes=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"\n[Benchmark] Running 1 epoch on {len(subset_ds)} images (batch_size={batch_size}, num_batches={len(loader)})...")
    batch_times = []
    t_start = time.perf_counter()

    for b_idx, (imgs, lbls) in enumerate(loader):
        t0 = time.perf_counter()
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)["loss"]
        loss.backward()
        optimizer.step()

        b_time = time.perf_counter() - t0
        batch_times.append(b_time)
        print(f"  Batch {b_idx + 1:2d}/{len(loader)}: loss={loss.item():.4f} | time={b_time * 1000:.1f} ms")

    total_time = time.perf_counter() - t_start
    mean_batch_time = np.mean(batch_times)
    median_batch_time = np.median(batch_times)

    print(f"\n{'='*55}")
    print(f"  CPU 200-image Benchmark Results (Batch size: {batch_size})")
    print(f"{'='*55}")
    print(f"  Total batches processed : {len(loader)}")
    print(f"  Total epoch time         : {total_time:.2f} s")
    print(f"  Mean time per batch      : {mean_batch_time * 1000:.1f} ms ({mean_batch_time:.3f} s)")
    print(f"  Median time per batch    : {median_batch_time * 1000:.1f} ms ({median_batch_time:.3f} s)")
    print(f"  Throughput               : {len(subset_ds) / total_time:.2f} images/s")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    generate_grid()
    benchmark_subset_run()

