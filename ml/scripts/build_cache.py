"""Pre-processing cache builder for APTOS 2019 retinal fundus images.

Operations:
1. Crop outer black circular borders / camera apertures (tol=15).
2. Resize to configurable square spatial resolution (default 512x512).
3. Save as un-normalized PNG to data/processed/cache_{size}/aptos/{id_code}.png.
4. Resumable: Skips already-cached valid files.
5. Progress bar and total cache size summary report.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinaguard_ml.data.preprocessing import crop_black_borders


def build_cache(
    raw_images_dir: Path,
    out_dir: Path,
    target_size: int = 512,
    tol: int = 15,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(list(raw_images_dir.glob("*.png")) + list(raw_images_dir.glob("*.jpg")))
    total_files = len(raw_files)

    if total_files == 0:
        print(f"[ERROR] No images found in {raw_images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Building Pre-Processed Fundus Cache ({target_size}x{target_size})")
    print(f"{'='*60}")
    print(f"  Source directory : {raw_images_dir}")
    print(f"  Output cache     : {out_dir}")
    print(f"  Total raw images : {total_files}")
    print(f"  Target size      : {target_size} x {target_size}")
    print(f"  Border tolerance : {tol}")
    print(f"{'='*60}\n")

    skipped = 0
    processed = 0

    for raw_p in tqdm(raw_files, desc=f"Caching {target_size}px", unit="img"):
        id_code = raw_p.stem
        target_p = out_dir / f"{id_code}.png"

        # Resumable skip check
        if target_p.exists() and target_p.stat().st_size > 1024:
            skipped += 1
            continue

        try:
            with Image.open(raw_p) as img:
                arr = np.array(img.convert("RGB"))
                cropped = crop_black_borders(arr, tol=tol)
                resized = Image.fromarray(cropped).resize((target_size, target_size), Image.Resampling.LANCZOS)
                resized.save(target_p, format="PNG")
                processed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {raw_p.name}: {e}", file=sys.stderr)

    # Compute cache size on disk
    cached_files = list(out_dir.glob("*.png"))
    total_bytes = sum(f.stat().st_size for f in cached_files)
    total_mb = total_bytes / (1024 * 1024)

    print(f"\n{'='*60}")
    print("  CACHE BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Newly processed  : {processed}")
    print(f"  Skipped (resumed): {skipped}")
    print(f"  Total in cache   : {len(cached_files)} images")
    print(f"  Total disk space : {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")
    print(f"  Target location  : {out_dir.resolve()}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cropped and resized fundus image cache.")
    parser.add_argument("--raw-dir", default="data/raw/aptos/train_images", help="Path to raw images dir")
    parser.add_argument("--size", type=int, default=512, help="Target square resolution (default: 512)")
    parser.add_argument("--out-dir", default=None, help="Output directory path (default: data/processed/cache_{size}/aptos)")
    parser.add_argument("--tol", type=int, default=15, help="Border crop threshold tolerance")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.out_dir is None:
        out_dir = Path(f"data/processed/cache_{args.size}/aptos")
    else:
        out_dir = Path(args.out_dir)

    build_cache(raw_dir, out_dir, target_size=args.size, tol=args.tol)


if __name__ == "__main__":
    main()

