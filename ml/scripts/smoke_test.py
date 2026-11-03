"""Smoke test: 2-epoch training on 200 stratified images + ONNX parity verification.

Validates the full ML training loop:
1. Loads 200 stratified fundus images from data/raw/aptos
2. Trains EfficientNetDRDualHead for 2 epochs using JointCategoricalOrdinalLoss
3. Verifies whether training loss decreases
4. Exports trained weights to ONNX format and validates numerical parity (tolerance <= 1e-4)
5. Measures and reports real elapsed execution time.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "backend" / "src"))
sys.path.insert(0, str(repo_root / "ml" / "src"))

from retinaguard_ml.export import export_to_onnx, verify_onnx_parity
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss
from retinaguard_ml.training.trainer import DRTrainer

FUNDUS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FUNDUS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class StratifiedAPTOSSubset(Dataset):
    """Loads a subset of real fundus photographs from disk with standard preprocessing."""

    def __init__(self, records: list[dict], raw_dir: Path, target_size: Tuple[int, int] = (224, 224)):
        self.records = records
        self.raw_dir = raw_dir
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = self.raw_dir / f"{rec['id_code']}.png"
        if not img_path.exists():
            img_path = self.raw_dir / f"{rec['id_code']}.jpg"

        with Image.open(img_path) as img:
            rgb = img.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32) / 255.0

        # Normalization
        arr = (arr - FUNDUS_MEAN) / FUNDUS_STD
        # HWC -> CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return tensor, rec["diagnosis"]


def run_smoke_test(
    splits_csv: Path,
    raw_dir: Path,
    num_samples: int = 200,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
) -> None:
    print(f"\n{'='*65}")
    print("  RetinaGuard ML Training Pipeline Smoke Test")
    print(f"{'='*65}")
    t0 = time.time()

    # 1. Load records and select 200 stratified samples
    records = []
    with open(splits_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "id_code": row["id_code"],
                "diagnosis": int(row["diagnosis"]),
                "split": row["split"],
            })

    # Group by grade
    by_grade: dict[int, list[dict]] = {g: [] for g in range(5)}
    for r in records:
        by_grade[r["diagnosis"]].append(r)

    # Stratified selection (proportional)
    rng = np.random.default_rng(seed=42)
    selected: list[dict] = []
    per_grade_target = num_samples // 5

    for g in range(5):
        available = by_grade[g]
        rng.shuffle(available)
        n_pick = min(len(available), per_grade_target)
        selected.extend(available[:n_pick])

    # Fill remainder if any
    if len(selected) < num_samples:
        rem_needed = num_samples - len(selected)
        pool = [r for g in range(5) for r in by_grade[g] if r not in selected]
        rng.shuffle(pool)
        selected.extend(pool[:rem_needed])

    print(f"[1/5] Loaded {len(selected)} stratified images across 5 grades:")
    counts = {g: sum(1 for r in selected if r["diagnosis"] == g) for g in range(5)}
    print(f"      {counts}")

    # 2. Build Dataset and DataLoader
    dataset = StratifiedAPTOSSubset(selected, raw_dir=raw_dir, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 3. Instantiate model, loss, optimizer
    print("\n[2/5] Initializing EfficientNetDRDualHead model and Joint Loss...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Using compute device: {device}")

    model = EfficientNetDRDualHead(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_classes=5,
    ).to(device)

    criterion = JointCategoricalOrdinalLoss(
        ordinal_weight=0.50,
        num_classes=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = DRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config={"training": {"mixed_precision": False, "gradient_clip_val": 1.0}},
        device=device,
    )

    # 4. Train 2 epochs
    print("\n[3/5] Executing 2 training epochs...")
    epoch_losses = []
    for epoch in range(1, 3):
        ep_t0 = time.time()
        metrics = trainer.train_epoch(dataloader)
        ep_time = time.time() - ep_t0
        loss = metrics["train_loss"]
        epoch_losses.append(loss)
        print(f"      Epoch {epoch}/2: train_loss={loss:.4f} (cat={metrics['train_cat_loss']:.4f}, ord={metrics['train_ord_loss']:.4f}) [{ep_time:.1f}s]")

    loss_decreased = epoch_losses[1] < epoch_losses[0]
    delta = epoch_losses[0] - epoch_losses[1]

    # 5. Export to ONNX and verify parity
    print("\n[4/5] Exporting model to ONNX & verifying parity...")
    export_dir = repo_root / "ml" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "retinaguard_smoke.onnx"

    input_shape = (1, 3, target_size[0], target_size[1])
    export_to_onnx(model, onnx_path, input_shape=input_shape)
    print(f"      ONNX export saved: {onnx_path}")

    parity_diffs = verify_onnx_parity(model, onnx_path, input_shape=input_shape, atol=1e-4)
    parity_passed = all(diff < 1e-4 for diff in parity_diffs.values())
    for out_name, max_diff in parity_diffs.items():
        print(f"      Output '{out_name}': max absolute difference = {max_diff:.2e}")

    total_time = time.time() - t0

    # 6. Final Results Summary
    print(f"\n{'='*65}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*65}")
    print(f"  Images used:               {len(selected)} (stratified)")
    print(f"  Epoch 1 Loss:              {epoch_losses[0]:.4f}")
    print(f"  Epoch 2 Loss:              {epoch_losses[1]:.4f}")
    print(f"  Loss Decreased:            {'YES' if loss_decreased else 'NO'} (delta: {delta:+.4f})")
    print(f"  ONNX Export Parity Test:   {'PASSED' if parity_passed else 'FAILED'}")
    print(f"  Total Elapsed Time:        {total_time:.2f} seconds")
    print(f"{'='*65}\n")


def main() -> None:
    splits_csv = Path("data/processed/splits_v1.csv")
    raw_dir = Path("data/raw/aptos/train_images")

    if not splits_csv.exists():
        print(f"[ERROR] Missing {splits_csv}. Run prepare_splits.py first.")
        sys.exit(1)
    if not raw_dir.exists():
        print(f"[ERROR] Missing {raw_dir}.")
        sys.exit(1)

    run_smoke_test(splits_csv, raw_dir)


if __name__ == "__main__":
    main()

