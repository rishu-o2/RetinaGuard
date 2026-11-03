#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import sys
import urllib.error

# Step 0: Pin KaggleHub cache directory inside the project workspace before import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "_kagglehub_cache"
os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR)

REQUIRED_FREE_SPACE_GB = 10.0  # Dataset download and extraction requires ~10 GB


def check_disk_space(target_path: Path, required_gb: float = REQUIRED_FREE_SPACE_GB) -> None:
    """Verify sufficient disk storage is available before initiating download."""
    target_path.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_path)
    free_gb = free / (1024**3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space on {target_path.drive or target_path}!\n"
            f"Available: {free_gb:.1f} GB. Required: {required_gb:.1f} GB minimum."
        )


def verify_kaggle_credentials() -> None:
    """Check whether Kaggle API credentials exist in environment or ~/.kaggle/kaggle.json."""
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()

    if not (has_env or has_file):
        raise RuntimeError(
            "Kaggle API credentials not found!\n"
            "Please configure your Kaggle credentials using either:\n"
            "  1. Place your API token at: ~/.kaggle/kaggle.json\n"
            "  2. Or export environment variables:\n"
            "       export KAGGLE_USERNAME='your_username'\n"
            "       export KAGGLE_KEY='your_api_key'\n"
            "To generate a token, visit: https://www.kaggle.com/settings -> 'Create New Token'."
        )


def download_and_extract_aptos(dest_dir: Path, force: bool = False) -> Path:
    """Download APTOS 2019 and copy train.csv and train_images/ to destination."""
    dest_dir = Path(dest_dir)
    train_csv = dest_dir / "train.csv"
    train_images = dest_dir / "train_images"

    # Idempotency check: Skip if already extracted unless --force is supplied
    if not force and train_csv.exists() and train_images.exists():
        img_count = len(list(train_images.glob("*.png")))
        if img_count > 0:
            print(f"[APTOS Downloader] Dataset already present at '{dest_dir}' ({img_count} images found).")
            print("Use --force to re-copy or overwrite.")
            return dest_dir

    # 1. Pre-flight checks
    check_disk_space(dest_dir)
    verify_kaggle_credentials()

    # 2. Import kagglehub dynamically after setting KAGGLEHUB_CACHE
    try:
        import kagglehub  # type: ignore
    except ImportError:
        raise RuntimeError(
            "kagglehub package is not installed.\n"
            "Please install dependencies with: pip install -r ml/requirements.txt"
        )

    print("[APTOS Downloader] Initiating download via kagglehub ('aptos2019-blindness-detection')...")
    try:
        download_path_str = kagglehub.competition_download("aptos2019-blindness-detection")
        download_path = Path(download_path_str)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                "HTTP 403 Forbidden: Competition rules have not been accepted!\n"
                "You must visit https://www.kaggle.com/c/aptos2019-blindness-detection/rules "
                "in your web browser, log in, and click 'I Understand and Accept' before downloading."
            ) from e
        raise RuntimeError(f"HTTP network error ({e.code}) while communicating with Kaggle: {e}") from e
    except Exception as e:
        err_msg = str(e).lower()
        if "403" in err_msg or "forbidden" in err_msg or "rules" in err_msg:
            raise RuntimeError(
                "Access Forbidden: Competition rules have not been accepted!\n"
                "Please visit https://www.kaggle.com/c/aptos2019-blindness-detection/rules "
                "and accept the rules with your Kaggle account."
            ) from e
        elif "credentials" in err_msg or "unauthorized" in err_msg or "401" in err_msg:
            raise RuntimeError("Invalid or expired Kaggle credentials. Please verify your API key.") from e
        raise RuntimeError(f"Download failed due to network or API error: {e}") from e

    print(f"[APTOS Downloader] Cache downloaded to: {download_path}")

    # 3. Copy only labeled training data
    # NOTE ON COMPETITION DATA:
    # We deliberately copy only 'train.csv' and 'train_images/'.
    # 'test.csv' and 'test_images/' are deliberately excluded because the Kaggle competition
    # test set ground-truth labels are held private by Kaggle and cannot be used for supervised
    # validation. For independent evaluation, RetinaGuard uses IDRiD as an external test set.
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_train_csv = download_path / "train.csv"
    src_train_images = download_path / "train_images"

    if not src_train_csv.exists():
        raise FileNotFoundError(f"Expected file 'train.csv' missing in downloaded cache: {download_path}")
    if not src_train_images.exists():
        raise FileNotFoundError(f"Expected folder 'train_images' missing in downloaded cache: {download_path}")

    print(f"[APTOS Downloader] Copying train.csv to {train_csv}...")
    shutil.copy2(src_train_csv, train_csv)

    print(f"[APTOS Downloader] Copying train_images/ to {train_images} (this may take a few moments)...")
    if train_images.exists() and force:
        shutil.rmtree(train_images)

    shutil.copytree(src_train_images, train_images, dirs_exist_ok=True)

    copied_count = len(list(train_images.glob("*.png")))
    print(f"[APTOS Downloader] Successfully populated {dest_dir} with {copied_count} training images.")
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare APTOS 2019 dataset for RetinaGuard.")
    parser.add_argument(
        "--dest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "aptos"),
        help="Destination directory (default: data/raw/aptos)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-copying even if destination already exists",
    )

    args = parser.parse_args()
    try:
        download_and_extract_aptos(dest_dir=Path(args.dest), force=args.force)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

