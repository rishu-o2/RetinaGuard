"""Script to fit and save In-Distribution Mahalanobis feature statistics.

Computes class centroids (mu_c) and pooled inverse covariance matrix (Sigma^-1).
Saves to configs/ood_statistics.npz for runtime OOD inference.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from retinaguard.core.config import _find_repo_root


def fit_in_distribution_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
    save_path: Optional[Path | str] = None,
    shrinkage: float = 1e-4,
) -> Path:
    """Fit class centroids and pooled precision matrix from training feature embeddings.

    Args:
        features: Penultimate feature vectors of shape [N, D].
        labels: Integer ground truth labels of shape [N].
        num_classes: Number of distinct classes (default 5 for ICDR).
        save_path: Output .npz path.
        shrinkage: Regularization constant added to covariance diagonal for numerical stability.

    Returns:
        Path to saved .npz file containing 'centroids' and 'inv_covariance'.
    """
    if save_path is None:
        save_path = _find_repo_root() / "configs" / "ood_statistics.npz"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, feature_dim = features.shape

    # 1. Compute class centroids
    centroids = np.zeros((num_classes, feature_dim), dtype=np.float32)
    centered_features = np.zeros_like(features)

    for c in range(num_classes):
        mask = (labels == c)
        if np.any(mask):
            class_feats = features[mask]
            centroid_c = np.mean(class_feats, axis=0)
            centroids[c] = centroid_c
            centered_features[mask] = class_feats - centroid_c
        else:
            # Fallback for unrepresented classes
            centroids[c] = np.mean(features, axis=0)

    # 2. Compute pooled covariance matrix: Sigma = (1 / N) * sum((z - mu_c)(z - mu_c)^T)
    covariance = np.dot(centered_features.T, centered_features) / float(n_samples)

    # Add Ledoit-Wolf / Tikhonov shrinkage to diagonal for invertibility
    covariance += shrinkage * np.eye(feature_dim, dtype=np.float32)

    # 3. Compute precision matrix (inverse covariance)
    inv_covariance = np.linalg.pinv(covariance).astype(np.float32)

    # 4. Save to .npz
    np.savez_compressed(
        str(save_path),
        centroids=centroids,
        inv_covariance=inv_covariance,
        num_classes=num_classes,
        feature_dim=feature_dim,
    )

    return save_path


if __name__ == "__main__":
    print("Generating baseline empirical OOD statistics for 512-dim features...")
    rng = np.random.default_rng(42)
    dummy_feats = rng.normal(loc=0.0, scale=1.0, size=(200, 512)).astype(np.float32)
    dummy_labels = rng.integers(0, 5, size=200)

    saved_file = fit_in_distribution_statistics(dummy_feats, dummy_labels)
    print(f"OOD statistics successfully saved to: {saved_file}")

