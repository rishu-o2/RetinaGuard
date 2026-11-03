"""Near-duplicate detection for APTOS 2019 via perceptual hashing.

APTOS 2019 provides NO patient IDs, so we cannot perform true patient-level
grouping. Instead we use:
  1. SHA-256 of raw file bytes  — catches exact byte-for-byte copies.
  2. Perceptual hash (pHash, 64-bit DCT-based)  — catches resized/re-compressed
     versions of the same underlying photograph.

Limitation (documented in ADR-015): this approach cannot detect two DIFFERENT
photographs of the SAME patient's other eye or a different visit. True patient-
level de-duplication requires the original clinical metadata which is not
publicly available for APTOS.

Outputs
-------
data/processed/duplicate_groups.csv
    Columns: group_id, id_code, diagnosis
    Every singleton image gets its own group_id (= its id_code).
    Images in the same near-duplicate cluster share the same group_id.

Printed to stdout:
  - Total images scanned
  - Number of exact duplicates (SHA-256 collision)
  - Number of near-duplicate groups (>1 member, pHash distance <= threshold)
  - Groups with CONFLICTING labels (label-noise candidates)
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _phash(path: Path, hash_size: int = 8):
    """Return imagehash pHash object for the image at path."""
    import imagehash  # type: ignore
    from PIL import Image  # type: ignore

    with Image.open(path) as img:
        return imagehash.phash(img, hash_size=hash_size)


def find_duplicates(
    images_dir: Path,
    csv_path: Path,
    hamming_threshold: int = 6,
    hash_size: int = 8,
) -> tuple[list[dict], int, int, list[list[dict]]]:
    """Scan images_dir and group near-duplicates.

    Returns
    -------
    records          : list of dicts {id_code, diagnosis, sha256, phash_hex, group_id}
    n_exact          : number of images that are exact byte duplicates of another
    n_near_dup_groups: number of groups with >1 member (including exact-dup groups)
    conflict_groups  : groups whose members have more than one distinct diagnosis label
    """
    import imagehash  # type: ignore

    # Read CSV
    id_to_grade: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_grade[row["id_code"].strip()] = int(row["diagnosis"].strip())

    records: list[dict] = []
    total = len(id_to_grade)
    print(f"  Hashing {total} images...", flush=True)

    for i, (id_code, grade) in enumerate(id_to_grade.items()):
        img_path = images_dir / f"{id_code}.png"
        if not img_path.exists():
            img_path = images_dir / f"{id_code}.jpg"
        if not img_path.exists():
            print(f"  [WARN] Missing: {id_code}", file=sys.stderr)
            continue

        sha = _sha256_file(img_path)
        ph = _phash(img_path, hash_size=hash_size)

        records.append({
            "id_code": id_code,
            "diagnosis": grade,
            "sha256": sha,
            "phash_hex": str(ph),
        })

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total}", flush=True)

    # ── Step 1: exact duplicates (SHA-256) ─────────────────────────────────
    sha_to_ids: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        sha_to_ids[rec["sha256"]].append(rec["id_code"])

    exact_dup_ids: set[str] = set()
    for ids in sha_to_ids.values():
        if len(ids) > 1:
            for id_code in ids[1:]:  # keep first, flag the rest
                exact_dup_ids.add(id_code)

    n_exact = len(exact_dup_ids)

    # ── Step 2: pHash clustering (union-find) ──────────────────────────────
    # Convert hex hashes to integer values for ultra-fast bitwise POPCNT comparison
    hash_ints = {
        rec["id_code"]: int(rec["phash_hex"], 16)
        for rec in records
    }

    # Union-Find
    parent: dict[str, str] = {rec["id_code"]: rec["id_code"] for rec in records}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids_list = list(hash_ints.keys())
    n_ids = len(ids_list)
    print(f"  Clustering {n_ids} images with Hamming distance <= {hamming_threshold}...", flush=True)

    for i in range(n_ids):
        id_a = ids_list[i]
        int_a = hash_ints[id_a]
        for j in range(i + 1, n_ids):
            id_b = ids_list[j]
            # Bitwise XOR count via Python's native int.bit_count() (hardware POPCNT)
            if (int_a ^ hash_ints[id_b]).bit_count() <= hamming_threshold:
                union(id_a, id_b)

    # Collect groups
    group_members: dict[str, list[str]] = defaultdict(list)
    for id_code in ids_list:
        group_members[find(id_code)].append(id_code)

    # Assign stable group_ids (root id_code as group_id)
    id_to_group: dict[str, str] = {}
    for root, members in group_members.items():
        for m in members:
            id_to_group[m] = root

    # Near-duplicate groups = groups with >1 member
    near_dup_groups = [m for m in group_members.values() if len(m) > 1]
    n_near_dup_groups = len(near_dup_groups)

    # ── Step 3: Label-conflict detection ──────────────────────────────────
    id_to_grade_map = {rec["id_code"]: rec["diagnosis"] for rec in records}
    conflict_groups: list[list[dict]] = []
    for members in near_dup_groups:
        grades = {id_to_grade_map[m] for m in members}
        if len(grades) > 1:
            conflict_groups.append([
                {"id_code": m, "diagnosis": id_to_grade_map[m]}
                for m in sorted(members)
            ])

    # Attach group_id to records
    for rec in records:
        rec["group_id"] = id_to_group[rec["id_code"]]

    return records, n_exact, n_near_dup_groups, conflict_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect near-duplicate images in APTOS 2019 via pHash."
    )
    parser.add_argument(
        "--root", default="data/raw/aptos",
        help="APTOS root dir containing train.csv and train_images/"
    )
    parser.add_argument(
        "--out", default="data/processed/duplicate_groups.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--threshold", type=int, default=6,
        help="Max pHash Hamming distance to consider near-duplicates (default: 6)"
    )
    parser.add_argument(
        "--hash-size", type=int, default=8,
        help="pHash grid size (default: 8 -> 64-bit hash)"
    )
    args = parser.parse_args()

    root = Path(args.root)
    images_dir = root / "train_images"
    csv_path = root / "train.csv"
    out_path = Path(args.out)

    for p in (images_dir, csv_path):
        if not p.exists():
            print(f"[ERROR] Not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"\n[find_duplicates] threshold={args.threshold}  hash_size={args.hash_size}x{args.hash_size}")
    records, n_exact, n_near_dup_groups, conflict_groups = find_duplicates(
        images_dir=images_dir,
        csv_path=csv_path,
        hamming_threshold=args.threshold,
        hash_size=args.hash_size,
    )

    # -- Save output CSV ----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "id_code", "diagnosis"])
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["group_id"], r["id_code"])):
            writer.writerow({
                "group_id": rec["group_id"],
                "id_code": rec["id_code"],
                "diagnosis": rec["diagnosis"],
            })

    # -- Print summary ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  APTOS 2019 Near-Duplicate Report")
    print(f"{'='*60}")
    print(f"  Total images scanned       : {len(records)}")
    print(f"  Exact duplicates (SHA-256) : {n_exact}")
    print(f"  Near-dup groups (pHash<={args.threshold}) : {n_near_dup_groups}")
    print(f"  Groups with CONFLICTING labels: {len(conflict_groups)}")

    if conflict_groups:
        print(f"\n  [!] LABEL-NOISE CANDIDATES (conflicting diagnoses within near-dup group):")
        for i, group in enumerate(conflict_groups):
            members_str = ", ".join(f"{m['id_code']}(grade={m['diagnosis']})" for m in group)
            print(f"    Group {i+1}: {members_str}")
    else:
        print("  [OK] No conflicting labels found within any near-duplicate group.")

    print(f"\n  Saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

