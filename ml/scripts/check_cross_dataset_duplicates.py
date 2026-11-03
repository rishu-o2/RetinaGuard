"""Cross-dataset duplicate check: IDRiD vs APTOS 2019.

Compares 64-bit DCT perceptual hashes (pHash) between the external test set (IDRiD)
and the training set (APTOS 2019) to detect any near-duplicate contamination.

Outputs:
- Prints any matches with Hamming distance <= threshold (default: 6)
- Saves data/processed/cross_dataset_duplicates.csv
- Does NOT delete or modify any images or manifests.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from PIL import Image
import imagehash


def compute_or_load_hashes(images_list: list[tuple[str, Path]], cache_path: Path) -> dict[str, int]:
    """Compute 64-bit pHash as int, caching results to disk as JSON."""
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    result_ints: dict[str, int] = {}
    dirty = False

    for img_id, path in images_list:
        if img_id in cache:
            result_ints[img_id] = int(cache[img_id], 16)
        else:
            try:
                with Image.open(path) as img:
                    ph = imagehash.phash(img, hash_size=8)
                    hex_str = str(ph)
                    cache[img_id] = hex_str
                    result_ints[img_id] = int(hex_str, 16)
                    dirty = True
            except Exception as e:
                print(f"[WARN] Failed to hash {path}: {e}", file=sys.stderr)

    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return result_ints


def check_cross_duplicates(
    aptos_csv: Path,
    aptos_img_dir: Path,
    idrid_manifest: Path,
    threshold: int = 6,
    out_csv: Path = Path("data/processed/cross_dataset_duplicates.csv"),
) -> list[dict]:
    print("=" * 65)
    print("       Cross-Dataset Contamination Check: IDRiD vs APTOS")
    print("=" * 65)

    # 1. Load APTOS items
    aptos_items = []
    with open(aptos_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_code = row["id_code"]
            p = aptos_img_dir / f"{id_code}.png"
            if not p.exists():
                p = aptos_img_dir / f"{id_code}.jpg"
            if p.exists():
                aptos_items.append((id_code, p))

    print(f"APTOS images: {len(aptos_items)}")
    aptos_cache = Path("data/processed/aptos_phash_cache.json")
    print("Hashing / loading APTOS pHashes...")
    aptos_hashes = compute_or_load_hashes(aptos_items, aptos_cache)

    # 2. Load IDRiD items
    idrid_items = []
    with open(idrid_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idrid_items.append((row["image_id"], Path(row["file_path"])))

    print(f"IDRiD images: {len(idrid_items)}")
    idrid_cache = Path("data/processed/idrid_phash_cache.json")
    print("Hashing / loading IDRiD pHashes...")
    idrid_hashes = compute_or_load_hashes(idrid_items, idrid_cache)

    # 3. Cross-compare
    print(f"\nComparing {len(idrid_hashes)} IDRiD against {len(aptos_hashes)} APTOS images (POPCNT <= {threshold})...")
    matches = []

    for idrid_id, idrid_int in idrid_hashes.items():
        for aptos_id, aptos_int in aptos_hashes.items():
            dist = (idrid_int ^ aptos_int).bit_count()
            if dist <= threshold:
                matches.append({
                    "idrid_id": idrid_id,
                    "aptos_id": aptos_id,
                    "hamming_distance": dist,
                })

    print("-" * 65)
    print(f"Total Cross-Dataset Near-Duplicate Pairs Found: {len(matches)}")
    print("-" * 65)

    if matches:
        print("[!] Contamination Warning: Found near-duplicates between IDRiD and APTOS:")
        for m in matches:
            print(f"  IDRiD: {m['idrid_id']} <-> APTOS: {m['aptos_id']} (Hamming dist: {m['hamming_distance']})")
    else:
        print("[+] CLEAN: Zero near-duplicates found between IDRiD and APTOS (threshold <= {}).".format(threshold))
        print("    IDRiD is confirmed strictly disjoint as an external test set.")

    # Save output CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idrid_id", "aptos_id", "hamming_distance"])
        writer.writeheader()
        writer.writerows(matches)

    print(f"\nSaved report to: {out_csv.resolve()}")
    print("=" * 65)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cross-dataset duplicates between IDRiD and APTOS.")
    parser.add_argument("--aptos-csv", default="data/raw/aptos/train.csv")
    parser.add_argument("--aptos-dir", default="data/raw/aptos/train_images")
    parser.add_argument("--idrid-manifest", default="data/processed/idrid_manifest.csv")
    parser.add_argument("--threshold", type=int, default=6)
    parser.add_argument("--out", default="data/processed/cross_dataset_duplicates.csv")
    args = parser.parse_args()

    check_cross_duplicates(
        aptos_csv=Path(args.aptos_csv),
        aptos_img_dir=Path(args.aptos_dir),
        idrid_manifest=Path(args.idrid_manifest),
        threshold=args.threshold,
        out_csv=Path(args.out),
    )


if __name__ == "__main__":
    main()

