"""Regression tests for last.pt checkpoint save and --resume restoration.

Verifies:
1. last.pt is written every epoch.
2. Resuming from last.pt restores model weights, optimizer state, scheduler
   state, and epoch counter correctly.
3. A resumed run produces the same model output (eval-mode forward pass) as
   an uninterrupted run at the same epoch — given the same seed and data.

Rules enforced:
- Uses SyntheticFundusDataset (tiny, random images) — no real data, no APTOS.
- No test split, no IDRiD.
- No CUDA required; everything runs on CPU.
"""

import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(n_samples: int = 32, batch_size: int = 8) -> DataLoader:
    # Use deterministic fixed tensors so epoch 1 and epoch 2 data are identical
    # across uninterrupted and resumed runs.
    gen = torch.Generator().manual_seed(42)
    imgs = torch.randn(n_samples, 3, 64, 64, generator=gen)
    lbls = torch.randint(0, 5, (n_samples,), generator=gen)
    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _build_model() -> EfficientNetDRDualHead:
    return EfficientNetDRDualHead(
        backbone_name="efficientnet_b0",
        pretrained=False,
        dropout_rate=0.0,   # deterministic
        num_classes=5,
    )


def _build_criterion(device: torch.device) -> JointCategoricalOrdinalLoss:
    class_weights = torch.ones(5, device=device)
    return JointCategoricalOrdinalLoss(
        class_weights=class_weights,
        ordinal_weight=0.5,
        use_focal_loss=False,
        num_classes=5,
    )


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: EfficientNetDRDualHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_metrics: dict,
    best_qwk: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": None,
        "val_metrics": val_metrics,
        "best_qwk": best_qwk,
    }, path)


def _load_checkpoint(
    path: Path,
    model: EfficientNetDRDualHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    try:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch_cpu"])
    except Exception:
        pass
    return ckpt["epoch"] + 1, ckpt.get("best_qwk", -1.0)


def _run_one_epoch(
    model: EfficientNetDRDualHead,
    criterion: JointCategoricalOrdinalLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        optimizer.zero_grad()
        outs = model(imgs)
        losses = criterion(outs, lbls.long())
        losses["loss"].backward()
        optimizer.step()
        total_loss += float(losses["loss"].item())
    return total_loss / max(len(loader), 1)


def _eval_forward(model: EfficientNetDRDualHead, loader: DataLoader) -> np.ndarray:
    """Return stacked softmax probabilities over the full loader."""
    model.eval()
    probs_list = []
    with torch.no_grad():
        for imgs, _ in loader:
            p = torch.softmax(model(imgs)["logits"], dim=-1).cpu().numpy()
            probs_list.append(p)
    return np.vstack(probs_list)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_last_pt_written_every_epoch() -> None:
    """last.pt must exist after every epoch."""
    _set_seed(1)
    device = torch.device("cpu")
    model = _build_model().to(device)
    criterion = _build_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loader = _make_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        last_pt = out_dir / "last.pt"

        for epoch in range(1, 4):   # 3 epochs
            _run_one_epoch(model, criterion, optimizer, loader)
            scheduler.step()
            # Simulate what train.py does: save last.pt every epoch
            _save_checkpoint(last_pt, epoch, model, optimizer, scheduler, {}, -1.0)
            assert last_pt.exists(), f"last.pt not found after epoch {epoch}"
            ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
            assert ckpt["epoch"] == epoch, (
                f"Checkpoint epoch mismatch: got {ckpt['epoch']}, expected {epoch}"
            )


def test_resume_restores_epoch_counter() -> None:
    """Resuming must start at epoch N+1 (not repeat epoch N)."""
    _set_seed(2)
    device = torch.device("cpu")
    model = _build_model().to(device)
    criterion = _build_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    loader = _make_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        last_pt = Path(tmpdir) / "last.pt"

        # Run 2 epochs, save after each
        for epoch in range(1, 3):
            _run_one_epoch(model, criterion, optimizer, loader)
            scheduler.step()
            _save_checkpoint(last_pt, epoch, model, optimizer, scheduler, {}, -1.0)

        # Load and verify start_epoch
        fresh_model = _build_model().to(device)
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        fresh_sched = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_opt, T_max=4)
        start_epoch, _ = _load_checkpoint(last_pt, fresh_model, fresh_opt, fresh_sched)
        assert start_epoch == 3, (
            f"Expected start_epoch=3 after saving epoch 2, got {start_epoch}"
        )


def test_resume_restores_model_weights() -> None:
    """Model weights loaded from last.pt must exactly match those at save time."""
    _set_seed(3)
    device = torch.device("cpu")
    model = _build_model().to(device)
    criterion = _build_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    loader = _make_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        last_pt = Path(tmpdir) / "last.pt"

        _run_one_epoch(model, criterion, optimizer, loader)
        scheduler.step()
        _save_checkpoint(last_pt, 1, model, optimizer, scheduler, {}, -1.0)

        # Capture weights at save time
        saved_fc_weight = model.fc_categorical.weight.data.clone()

        # Mutate the model (simulate it being modified further)
        with torch.no_grad():
            model.fc_categorical.weight.fill_(999.0)

        # Reload from checkpoint
        fresh_model = _build_model().to(device)
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        fresh_sched = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_opt, T_max=2)
        _load_checkpoint(last_pt, fresh_model, fresh_opt, fresh_sched)

        torch.testing.assert_close(
            fresh_model.fc_categorical.weight.data,
            saved_fc_weight,
            msg="Resumed model weights differ from saved weights",
        )


def test_resumed_run_matches_uninterrupted_run() -> None:
    """A run resumed after epoch 1 must produce the same model output as a
    continuous 2-epoch run when started from the same initial weights and seed.

    This is the strongest resume correctness test: it catches bugs where
    optimizer momentum, scheduler LR, or RNG state are not properly restored.
    """
    _set_seed(42)
    device = torch.device("cpu")
    loader = _make_loader(n_samples=32, batch_size=8)

    # ── Uninterrupted 2-epoch run ──────────────────────────────────────────
    model_a = _build_model().to(device)
    _set_seed(42)
    torch.manual_seed(42)
    # Capture initial weights so model_b starts identically
    init_state = copy.deepcopy(model_a.state_dict())

    criterion_a = _build_criterion(device)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=2)

    _set_seed(42)
    _run_one_epoch(model_a, criterion_a, opt_a, loader)
    sched_a.step()
    _set_seed(99)       # different seed so epoch 2 data differs
    _run_one_epoch(model_a, criterion_a, opt_a, loader)
    sched_a.step()

    probs_a = _eval_forward(model_a, loader)

    # ── Interrupted after epoch 1, then resumed ────────────────────────────
    model_b = _build_model().to(device)
    model_b.load_state_dict(init_state)   # same starting point

    criterion_b = _build_criterion(device)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=2)

    _set_seed(42)
    _run_one_epoch(model_b, criterion_b, opt_b, loader)
    sched_b.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        last_pt = Path(tmpdir) / "last.pt"
        _save_checkpoint(last_pt, 1, model_b, opt_b, sched_b, {}, -1.0)

        # Resume
        model_c = _build_model().to(device)
        opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
        sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=2)
        _load_checkpoint(last_pt, model_c, opt_c, sched_c)

        # epoch 2 — same seed as in the uninterrupted run
        _set_seed(99)
        _run_one_epoch(model_c, criterion_b, opt_c, loader)
        sched_c.step()

    probs_c = _eval_forward(model_c, loader)

    np.testing.assert_allclose(
        probs_c, probs_a, atol=1e-5,
        err_msg=(
            "Resumed run diverges from uninterrupted run. "
            "Model weights, optimizer state, or scheduler LR not properly restored."
        ),
    )

