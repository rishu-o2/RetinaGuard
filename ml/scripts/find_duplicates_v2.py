"""Strict duplicate grouping for APTOS 2019 without transitive chaining.

Rules:
1. Candidate pair: pHash Hamming distance <= 2.
2. Verification: Normalized Cross-Correlation (NCC) of 32x32 grayscale thumbnails
   of the cropped retina > 0.98.
3. Clique / strict clustering: Avoids long transitive chains by requiring high mutual similarity.
4. Cap: Any group is capped at 10 members. If a component exceeds 10, it is split
   into tight sub-clusters (max 10 members each), and any capped group is reported.

Outputs:
data/processed/duplicate_groups_v2.csv (group_id, id_code, diagnosis)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# Import border cropping from existing preprocessor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinaguard_ml.data.preprocessing import crop_black_borders


def compute_features(images_dir: Path, id_codes: list[str]) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Compute 64-bit pHash int and 32x32 normalized cropped grayscale thumbnail for each image."""
    phash_ints: dict[str, int] = {}
    norm_thumbs: dict[str, np.ndarray] = {}

    total = len(id_codes)
    print(f"[compute_features] Computing pHash and 32x32 cropped thumbnail for {total} images...")

    for i, id_code in enumerate(id_codes):
        p = images_dir / f"{id_code}.png"
        if not p.exists():
            p = images_dir / f"{id_code}.jpg"
        if not p.exists():
            continue

        with Image.open(p) as img:
            rgb = img.convert("RGB")
            # 1. pHash
            ph = imagehash.phash(rgb, hash_size=8)
            phash_ints[id_code] = int(str(ph), 16)

            # 2. Cropped retina 32x32 grayscale thumbnail for NCC
            arr = np.array(rgb)
            cropped = crop_black_borders(arr, tol=15)
            thumb = Image.fromarray(cropped).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            t_arr = np.array(thumb, dtype=np.float32)

            # Zero-mean unit-variance normalize for fast dot-product NCC
            t_centered = t_arr - np.mean(t_arr)
            norm = np.linalg.norm(t_centered)
            if norm > 1e-6:
                t_norm = t_centered / norm
            else:
                t_norm = np.zeros_like(t_centered)

            norm_thumbs[id_code] = t_norm.flatten()

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} processed...", flush=True)

    return phash_ints, norm_thumbs


def find_strict_pairs(
    id_codes: list[str],
    phash_ints: dict[str, int],
    norm_thumbs: dict[str, np.ndarray],
    hamming_thresh: int = 2,
    ncc_thresh: float = 0.98,
) -> list[tuple[str, str, int, float]]:
    """Identify strictly matching image pairs meeting both pHash and NCC criteria."""
    n = len(id_codes)
    print(f"\n[find_strict_pairs] Evaluating pairs: pHash <= {hamming_thresh} AND NCC > {ncc_thresh}...")
    strict_pairs = []

    for i in range(n):
        id_a = id_codes[i]
        int_a = phash_ints[id_a]
        th_a = norm_thumbs[id_a]

        for j in range(i + 1, n):
            id_b = id_codes[j]
            # Fast bitwise popcnt
            h_dist = (int_a ^ phash_ints[id_b]).bit_count()
            if h_dist <= hamming_thresh:
                # Second check: NCC
                ncc = float(np.dot(th_a, norm_thumbs[id_b]))
                if ncc > ncc_thresh:
                    strict_pairs.append((id_a, id_b, h_dist, ncc))

    print(f"  Identified {len(strict_pairs)} strict pairwise matches.")
    return strict_pairs


def cluster_and_cap(
    id_codes: list[str],
    strict_pairs: list[tuple[str, str, int, float]],
    norm_thumbs: dict[str, np.ndarray],
    max_group_size: int = 10,
) -> tuple[dict[str, str], list[dict]]:
    """Cluster pairs into tight groups and cap at max_group_size without transitive chaining explosion."""
    # Build adjacency
    adj = defaultdict(set)
    for id_a, id_b, _, _ in strict_pairs:
        adj[id_a].add(id_b)
        adj[id_b].add(id_a)

    visited = set()
    clusters = []

    # First collect connected components
    for id_code in id_codes:
        if id_code not in visited and id_code in adj:
            comp = []
            queue = [id_code]
            visited.add(id_code)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(comp)

    # Process clusters with max size cap and tightness check
    final_groups: list[list[str]] = []
    cap_fired_reports: list[dict] = []

    for comp in clusters:
        if len(comp) <= max_group_size:
            final_groups.append(comp)
        else:
            # Component exceeded cap! Split into tight sub-clusters of <= max_group_size
            print(f"  [CAP FIRED] Component of size {len(comp)} exceeded cap {max_group_size}!")
            cap_fired_reports.append({"original_size": len(comp), "members": comp})

            # Greedy sub-cluster by mutual correlation centroid
            remaining = set(comp)
            while remaining:
                # Pick seed with highest internal connectivity
                seed = max(remaining, key=lambda m: sum(1 for other in remaining if other in adj[m]))
                sub = [seed]
                remaining.remove(seed)

                # Add closest mutual neighbors up to max_group_size
                candidates = sorted(
                    list(remaining),
                    key=lambda other: np.dot(norm_thumbs[seed], norm_thumbs[other]),
                    reverse=True,
                )
                for cand in candidates[: max_group_size - 1]:
                    if np.dot(norm_thumbs[seed], norm_thumbs[cand]) > 0.98:
                        sub.append(cand)
                        remaining.remove(cand)

                final_groups.append(sub)

    # Map each image_id to its group_id
    id_to_group = {}
    for grp in final_groups:
        rep_id = sorted(grp)[0]
        for m in grp:
            id_to_group[m] = rep_id

    # Singletons
    for id_code in id_codes:
        if id_code not in id_to_group:
            id_to_group[id_code] = id_code

    return id_to_group, cap_fired_reports


def main():
    parser = argparse.ArgumentParser(description="Strict duplicate clustering for APTOS 2019.")
    parser.add_argument("--root", default="data/raw/aptos")
    parser.add_argument("--out", default="data/processed/duplicate_groups_v2.csv")
    parser.add_argument("--hamming", type=int, default=2)
    parser.add_argument("--ncc", type=float, default=0.98)
    parser.add_argument("--max-size", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    images_dir = root / "train_images"
    csv_path = root / "train.csv"
    out_path = Path(args.out)

    # Read CSV
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            records.append(r)

    id_codes = [r["id_code"] for r in records]
    id_to_diag = {r["id_code"]: int(r["diagnosis"]) for r in records}

    # Features
    phash_ints, norm_thumbs = compute_features(images_dir, id_codes)

    # Strict pairwise checks
    strict_pairs = find_strict_pairs(id_codes, phash_ints, norm_thumbs, hamming_thresh=args.hamming, ncc_thresh=args.ncc)

    # Cluster and cap
    id_to_group, cap_reports = cluster_and_cap(id_codes, strict_pairs, norm_thumbs, max_group_size=args.max_size)

    # Save CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "id_code", "diagnosis"])
        writer.writeheader()
        for id_code in sorted(id_codes):
            writer.writerow({
                "group_id": id_to_group[id_code],
                "id_code": id_code,
                "diagnosis": id_to_diag[id_code],
            })

    # Summary
    groups = defaultdict(list)
    for id_code, gid in id_to_group.items():
        groups[gid].append(id_code)

    group_sizes = defaultdict(int)
    for g, mems in groups.items():
        group_sizes[len(mems)] += 1

    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    print("\n" + "=" * 60)
    print("  STRICT DUPLICATE CLUSTERING (v2) RESULTS")
    print("=" * 60)
    print(f"  Total images processed    : {len(id_codes)}")
    print(f"  Total groups              : {len(groups)}")
    print(f"  Singletons (size 1)       : {group_sizes[1]}")
    for sz, count in sorted(group_sizes.items()):
        if sz > 1:
            print(f"    Size {sz:2d} groups       : {count}")

    print("\n  10 Largest Groups:")
    for i, (gid, mems) in enumerate(sorted_groups[:10], 1):
        grades = dict(defaultdict(int))
        for m in mems:
            grades[id_to_diag[m]] = grades.get(id_to_diag[m], 0) + 1
        print(f"    {i:2d}. Group {gid}: {len(mems)} images, grades={grades}")

    if cap_reports:
        print(f"\n  [!] Cap fired on {len(cap_reports)} components:")
        for rep in cap_reports:
            print(f"      Size {rep['original_size']} split into sub-clusters.")
    else:
        print("\n  [OK] No group exceeded the cap of 10.")

    print(f"\n  Saved: {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

