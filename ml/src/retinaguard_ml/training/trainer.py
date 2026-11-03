"""RetinaGuard Model Training Engine.

Features:
- Config-driven training loop (YAML).
- Mixed precision (torch.amp) & gradient clipping.
- Joint loss optimization (Categorical + Ordinal regression).
- Comprehensive validation metrics (QWK, Sensitivity @ 90%/95% Specificity, AUC).
- Checkpointing & early stopping based on validation QWK.
- Pluggable experiment logging (file/JSONL, MLflow, Weights & Biases).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from retinaguard_ml.evaluate import evaluate_cohort
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss


class ExperimentLogger:
    """Unified experiment logger supporting local JSONL files, W&B, and MLflow."""

    def __init__(
        self,
        logger_type: str = "file",
        log_dir: str = "logs/training",
        project_name: str = "retinaguard-dr",
        experiment_name: str = "run",
    ):
        self.logger_type = logger_type.lower()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self._wandb = None
        self._mlflow = None

        if self.logger_type == "wandb":
            try:
                import wandb  # type: ignore
                self._wandb = wandb
                wandb.init(project=project_name, name=experiment_name)
            except ImportError:
                print("Weights & Biases not installed. Falling back to file logger.")
                self.logger_type = "file"
        elif self.logger_type == "mlflow":
            try:
                import mlflow  # type: ignore
                self._mlflow = mlflow
                mlflow.set_experiment(project_name)
                mlflow.start_run(run_name=experiment_name)
            except ImportError:
                print("MLflow not installed. Falling back to file logger.")
                self.logger_type = "file"

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log numeric metrics for the current epoch/step."""
        # Convert non-serializable objects (numpy, lists)
        clean_metrics: Dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                clean_metrics[k] = v
            elif isinstance(v, (np.floating, np.integer)):
                clean_metrics[k] = float(v)

        # Write to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_metrics) + "\n")

        if self.logger_type == "wandb" and self._wandb:
            self._wandb.log(clean_metrics, step=step)
        elif self.logger_type == "mlflow" and self._mlflow:
            for k, v in clean_metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, v, step=step)


class DRTrainer:
    """Trainer orchestrator for Diabetic Retinopathy Dual-Head models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training parameters
        train_cfg = self.config.get("training", {})
        self.mixed_precision = train_cfg.get("mixed_precision", True) and (self.device.type == "cuda")
        self.grad_clip_val = train_cfg.get("gradient_clip_val", 1.0)
        self.patience = train_cfg.get("early_stopping_patience", 6)
        self.save_dir = Path(train_cfg.get("save_dir", "models/checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        # Setup experiment logger
        log_cfg = self.config.get("logging", {})
        self.logger = ExperimentLogger(
            logger_type=log_cfg.get("logger_type", "file"),
            log_dir=log_cfg.get("log_dir", "logs/training"),
            project_name=log_cfg.get("project_name", "retinaguard-dr"),
            experiment_name=log_cfg.get("experiment_name", "run"),
        )

        # Tracking state
        self.best_qwk = -1.0
        self.patience_counter = 0

    @classmethod
    def from_config_file(
        cls, config_path: Path | str, class_weights: Optional[torch.Tensor] = None
    ) -> "DRTrainer":
        """Instantiate DRTrainer from YAML configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        m_cfg = cfg.get("model", {})
        model = EfficientNetDRDualHead(
            backbone_name=m_cfg.get("backbone", "efficientnet_b0"),
            pretrained=m_cfg.get("pretrained", True),
            dropout_rate=m_cfg.get("dropout_rate", 0.30),
            num_classes=m_cfg.get("num_classes", 5),
        )

        l_cfg = cfg.get("loss", {})
        criterion = JointCategoricalOrdinalLoss(
            class_weights=class_weights,
            ordinal_weight=l_cfg.get("ordinal_loss_weight", 0.50),
            use_focal_loss=(l_cfg.get("categorical_loss") == "focal"),
            focal_gamma=l_cfg.get("focal_gamma", 2.0),
            num_classes=m_cfg.get("num_classes", 5),
        )

        opt_cfg = cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("learning_rate", 3e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
        )

        sched_cfg = cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 25),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

        return cls(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, config=cfg)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch."""
        # External holdout safety check
        ds = getattr(dataloader, "dataset", None)
        if ds is not None:
            ds_name = getattr(ds, "dataset_name", None)
            from retinaguard_ml.data.registry import DATASET_REGISTRY
            if ds_name and DATASET_REGISTRY.get(ds_name) and DATASET_REGISTRY[ds_name].is_external_test_only:
                from retinaguard_ml.data.split import ExternalHoldoutContaminationError
                raise ExternalHoldoutContaminationError(
                    f"Dataset '{ds_name}' is marked external_test_only and cannot be used for training!"
                )

        self.model.train()
        total_loss = 0.0
        total_cat_loss = 0.0
        total_ord_loss = 0.0
        num_batches = len(dataloader)

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.grad_clip_val > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_cat_loss += float(loss_dict["categorical_loss"].item())
            total_ord_loss += float(loss_dict["ordinal_loss"].item())

        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_cat_loss": total_cat_loss / max(num_batches, 1),
            "train_ord_loss": total_ord_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, labels)
                total_loss += float(loss_dict["loss"].item())

            # Probabilities and predictions
            cat_probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            ord_probs = torch.sigmoid(outputs["ordinal_logits"])
            expected_grades = torch.round(torch.sum(ord_probs, dim=-1)).long().clamp(0, 4)

            all_preds.extend(expected_grades.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.append(cat_probs)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.vstack(all_probs) if all_probs else None

        metrics = evaluate_cohort(y_true, y_pred, y_probs=y_probs)
        metrics["val_loss"] = total_loss / max(len(dataloader), 1)

        return metrics

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]) -> Path:
        """Save training state checkpoint."""
        save_path = self.save_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        return save_path

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> Dict[str, Any]:
        """Run full training lifecycle with checkpointing and early stopping."""
        history: List[Dict[str, Any]] = []

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            # Merge metrics
            epoch_summary = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(epoch_summary)
            self.logger.log_metrics(epoch_summary, step=epoch)

            # Check for best validation QWK
            current_qwk = val_metrics.get("qwk", -1.0)
            if current_qwk > self.best_qwk:
                self.best_qwk = current_qwk
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
            else:
                self.patience_counter += 1

            self.save_checkpoint("latest_checkpoint.pt", epoch, val_metrics)

            # Early stopping check
            if self.patience_counter >= self.patience:
                break

        return {
            "best_qwk": self.best_qwk,
            "history": history,
            "stopped_early": self.patience_counter >= self.patience,
        }

