"""RetinaGuard Production Training Script.

Trains EfficientNet Dual-Head on train split, validates on val split each epoch,
and NEVER touches the test split.

Features:
- Checkpoints last.pt and best.pt (by validation QWK).
- Full state restoration via --resume (model, optimizer, scheduler, scaler, epoch, RNG).
- Per-epoch metrics logged to {out_dir}/metrics.csv.
- Early stopping based on validation QWK.
- Out-dir path flexible for local or Google Drive mounting.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# Add module paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinaguard_ml.data.loaders.aptos import APTOSDataset
from retinaguard_ml.evaluate import evaluate_cohort
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
    val_metrics: Dict[str, Any],
    best_qwk: float,
) -> None:
    """Save full training state dictionary including RNG states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "val_metrics": val_metrics,
        "best_qwk": best_qwk,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> tuple[int, float]:
    """Restore model, optimizer, scheduler, scaler, and RNG states."""
    print(f"\n[--resume] Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore RNG
    try:
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch_cpu"])
        if torch.cuda.is_available() and checkpoint.get("rng_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_torch_cuda"])
    except Exception as e:
        print(f"[WARN] Could not restore all RNG states: {e}")

    start_epoch = checkpoint["epoch"] + 1
    best_qwk = checkpoint.get("best_qwk", -1.0)
    print(f"[--resume] Resumed from epoch {checkpoint['epoch']} (Best QWK: {best_qwk:.4f})\n")
    return start_epoch, best_qwk


def train(
    config_path: Path,
    cache_dir: Path,
    splits_path: Path,
    out_dir: Path,
    epochs_override: Optional[int] = None,
    resume_path: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Deterministic Seed
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  RetinaGuard Production Training Loop")
    print(f"{'='*65}")
    print(f"  Device           : {device}")
    print(f"  Config           : {config_path}")
    print(f"  Splits manifest  : {splits_path}")
    print(f"  Cache dir        : {cache_dir}")
    print(f"  Output directory : {out_dir.resolve()}")
    print("  Partition Safety : Strictly Training & Validation. NEVER TOUCHES TEST SPLIT.")
    print(f"{'='*65}\n")

    # 1. Datasets and Loaders
    train_dataset = APTOSDataset(
        manifest_path=splits_path,
        cache_dir=cache_dir,
        split="train",
        augment=True,
    )
    val_dataset = APTOSDataset(
        manifest_path=splits_path,
        cache_dir=cache_dir,
        split="val",
        augment=False,
    )

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = cfg.get("training", {}).get("num_workers", 2)

    # Imbalance handling: Weighted sampler
    sampler = train_dataset.get_weighted_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Data] Train set size: {len(train_dataset)} | Val set size: {len(val_dataset)}")

    # 2. Model & Loss
    m_cfg = cfg.get("model", {})
    model = EfficientNetDRDualHead(
        backbone_name=m_cfg.get("backbone", "efficientnet_b0"),
        pretrained=m_cfg.get("pretrained", True),
        dropout_rate=m_cfg.get("dropout_rate", 0.30),
        num_classes=m_cfg.get("num_classes", 5),
    ).to(device)

    l_cfg = cfg.get("loss", {})
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = JointCategoricalOrdinalLoss(
        class_weights=class_weights,
        ordinal_weight=l_cfg.get("ordinal_loss_weight", 0.50),
        use_focal_loss=(l_cfg.get("categorical_loss") == "focal"),
        focal_gamma=l_cfg.get("focal_gamma", 2.0),
        num_classes=5,
    ).to(device)

    # 3. Optimizer & Scheduler
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("learning_rate", 3e-4),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    total_epochs = epochs_override if epochs_override is not None else cfg.get("training", {}).get("epochs", 25)
    sched_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg.get("T_max", total_epochs),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    mixed_precision = cfg.get("training", {}).get("mixed_precision", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    grad_clip_val = cfg.get("training", {}).get("gradient_clip_val", 1.0)
    patience = cfg.get("training", {}).get("early_stopping_patience", 6)

    # 4. Resume check
    start_epoch = 1
    best_qwk = -1.0
    if resume_path and Path(resume_path).exists():
        start_epoch, best_qwk = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, scaler, device
        )

    # 5. Metrics CSV Initialization
    metrics_csv_path = out_dir / "metrics.csv"
    csv_exists = metrics_csv_path.exists()
    metrics_file = open(metrics_csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(metrics_file)
    if not csv_exists:
        writer.writerow([
            "epoch",
            "train_loss",
            "train_cat_loss",
            "train_ord_loss",
            "val_loss",
            "val_qwk",
            "val_referable_sens",
            "val_referable_spec",
            "val_auc",
            "lr",
            "elapsed_s",
        ])
        metrics_file.flush()

    patience_counter = 0

    # 6. Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_cat_loss = 0.0
        train_ord_loss = 0.0
        n_train_batches = len(train_loader)

        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
                outs = model(imgs)
                losses = criterion(outs, lbls)
                loss = losses["loss"]

            scaler.scale(loss).backward()
            if grad_clip_val > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_cat_loss += float(losses["categorical_loss"].item())
            train_ord_loss += float(losses["ordinal_loss"].item())

        train_loss /= max(n_train_batches, 1)
        train_cat_loss /= max(n_train_batches, 1)
        train_ord_loss /= max(n_train_batches, 1)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
                    outs = model(imgs)
                    losses = criterion(outs, lbls)
                    val_loss += float(losses["loss"].item())

                # Output parsing
                cat_probs = torch.softmax(outs["logits"], dim=-1).cpu().numpy()
                ord_probs = torch.sigmoid(outs["ordinal_logits"])
                preds = torch.round(torch.sum(ord_probs, dim=-1)).long().clamp(0, 4)

                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(lbls.cpu().numpy().tolist())
                val_probs.append(cat_probs)

        val_loss /= max(len(val_loader), 1)
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Calculate Validation Metrics
        eval_metrics = evaluate_cohort(
            np.array(val_targets),
            np.array(val_preds),
            y_probs=np.vstack(val_probs) if val_probs else None,
        )
        # Use direct indexing so a missing key raises KeyError immediately
        # (cannot silently become 0.0 the way .get(..., 0.0) would).
        val_qwk  = eval_metrics["qwk"]
        ref_sens = eval_metrics["sensitivity"]
        ref_spec = eval_metrics["specificity"]
        # referable_dr_auc is only present when y_probs was supplied; fall back
        # to 0.0 only in the (impossible during training) case it is absent.
        auc      = eval_metrics.get("referable_dr_auc", 0.0)

        ep_duration = time.time() - ep_t0

        # Log row to CSV
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_cat_loss:.4f}",
            f"{train_ord_loss:.4f}",
            f"{val_loss:.4f}",
            f"{val_qwk:.4f}",
            f"{ref_sens:.4f}",
            f"{ref_spec:.4f}",
            f"{auc:.4f}",
            f"{curr_lr:.6f}",
            f"{ep_duration:.1f}",
        ])
        metrics_file.flush()

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} [{ep_duration:4.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f} | "
            f"Ref Sens: {ref_sens:.3f} | Ref Spec: {ref_spec:.3f} | "
            f"LR: {curr_lr:.6f}"
        )

        # Checkpointing
        save_checkpoint(
            out_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            eval_metrics,
            best_qwk,
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0
            save_checkpoint(
                out_dir / "best.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                eval_metrics,
                best_qwk,
            )
            print(f"  [+] New best model saved! (QWK: {best_qwk:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered: No QWK improvement for {patience} consecutive epochs.")
                break

    metrics_file.close()
    print(f"\n[+] Training complete. Best Validation QWK: {best_qwk:.4f}")
    print(f"    Checkpoints and metrics saved in: {out_dir.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaGuard Model Training")
    parser.add_argument("--config", default="configs/baseline_b0_384.yaml", help="Path to YAML training config")
    parser.add_argument("--cache-dir", default="data/processed/cache_512/aptos", help="Path to cropped/cached image dir")
    parser.add_argument("--splits", default="data/processed/splits_v2.csv", help="Path to splits manifest CSV")
    parser.add_argument("--out-dir", default="models/checkpoints/baseline_b0_384", help="Checkpoint and metric output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
    args = parser.parse_args()

    train(
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        splits_path=Path(args.splits),
        out_dir=Path(args.out_dir),
        epochs_override=args.epochs,
        resume_path=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()

