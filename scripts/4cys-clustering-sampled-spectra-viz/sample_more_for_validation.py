#!/usr/bin/env python3
"""Sample additional structures per cluster for a second validation round.

For each k-means cluster in ``approach1/labels.csv`` this samples a number of
*new* structures (default 3, but see ``--cluster-overrides``), copies their
aligned ``.xyz`` files from ``aligned_xyz`` into a fresh "round II" directory,
and guarantees:

  * No overlap with structures already sampled in the round-I directory
    (``sampled-xyz-files-for-val``).
  * The total sampled fraction of any cluster (round I + round II combined)
    never exceeds ``--max-fraction`` (default 0.5).  So a cluster with n=6 and
    2 already sampled can receive at most 1 more (floor(6 * 0.5) = 3 total).

Run with the project environment:

    uv run scripts/4cys-clustering-sampled-spectra-viz/sample_more_for_validation.py

Use ``--dry-run`` to preview the selection without copying anything.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPROACH = REPO_ROOT / "data/test-4cys-weighted/approach1"
DEFAULT_LABELS = DEFAULT_APPROACH / "labels.csv"
DEFAULT_ALIGNED = DEFAULT_APPROACH / "aligned_xyz"
DEFAULT_EXISTING = (
    REPO_ROOT
    / "data/test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "data/test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val-II"
)


def parse_overrides(spec: str | None) -> dict[int, int]:
    """Parse ``"20:8,4:2"`` into ``{20: 8, 4: 2}``."""
    out: dict[int, int] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cluster_str, count_str = part.split(":")
        out[int(cluster_str)] = int(count_str)
    return out


def load_clusters(labels_csv: Path) -> dict[int, list[str]]:
    """Return ``{cluster: [structure_id, ...]}`` from labels.csv."""
    members: dict[int, list[str]] = defaultdict(list)
    with labels_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            members[int(row["cluster"])].append(row["structure_id"])
    return dict(members)


def existing_stems(existing_dir: Path) -> set[str]:
    """Structure ids already sampled (by .xyz stem) in the round-I directory."""
    if not existing_dir.is_dir():
        return set()
    return {p.stem for p in existing_dir.glob("*.xyz")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--aligned-xyz", type=Path, default=DEFAULT_ALIGNED)
    ap.add_argument("--existing", type=Path, default=DEFAULT_EXISTING,
                    help="round-I sampled dir; new picks must not overlap it")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="round-II output dir for the newly sampled .xyz files")
    ap.add_argument("--n-per-cluster", type=int, default=3,
                    help="target number of NEW structures per cluster")
    ap.add_argument("--cluster-overrides", default="20:8",
                    help='per-cluster target overrides, e.g. "20:8,4:2"')
    ap.add_argument("--max-fraction", type=float, default=0.5,
                    help="max fraction of a cluster sampled in total (I + II)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection but copy nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    overrides = parse_overrides(args.cluster_overrides)

    clusters = load_clusters(args.labels)
    already = existing_stems(args.existing)

    print(f"labels:        {args.labels}")
    print(f"aligned xyz:   {args.aligned_xyz}")
    print(f"existing (I):  {args.existing}  ({len(already)} files)")
    print(f"output  (II):  {args.out}")
    print(f"seed={args.seed}  n_per_cluster={args.n_per_cluster}  "
          f"overrides={overrides}  max_fraction={args.max_fraction}")
    print("-" * 88)
    header = (f"{'clust':>5}  {'n':>5}  {'cap':>4}  {'had':>4}  "
              f"{'want':>4}  {'room':>4}  {'avail':>5}  {'add':>4}  note")
    print(header)

    selected: dict[int, list[str]] = {}
    total_add = 0
    warnings: list[str] = []

    for cluster in sorted(clusters):
        members = clusters[cluster]
        n = len(members)
        cap = math.floor(n * args.max_fraction)          # max total sampled
        had = sum(1 for m in members if m in already)     # round-I in this cluster
        want = overrides.get(cluster, args.n_per_cluster)
        room = max(0, cap - had)                           # 50%-cap headroom
        available = [m for m in members if m not in already]
        n_add = min(want, room, len(available))

        note = ""
        if n_add < want:
            reasons = []
            if room < want:
                reasons.append(f"capped@50% (cap={cap}, had={had})")
            if len(available) < want:
                reasons.append(f"only {len(available)} unsampled left")
            note = "; ".join(reasons)
            warnings.append(f"cluster {cluster}: wanted {want}, added {n_add} "
                            f"({note})")

        picks = rng.sample(available, n_add) if n_add else []
        selected[cluster] = picks
        total_add += n_add

        print(f"{cluster:>5}  {n:>5}  {cap:>4}  {had:>4}  {want:>4}  "
              f"{room:>4}  {len(available):>5}  {n_add:>4}  {note}")

    print("-" * 88)
    print(f"total new structures to sample: {total_add}")
    if warnings:
        print("\nnotes:")
        for w in warnings:
            print(f"  - {w}")

    # -- copy --------------------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] no files copied.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing_src: list[str] = []
    for cluster in sorted(selected):
        for sid in selected[cluster]:
            src = args.aligned_xyz / f"{sid}.xyz"
            if not src.is_file():
                missing_src.append(sid)
                continue
            shutil.copy2(src, args.out / src.name)
            copied += 1

    print(f"\ncopied {copied} files -> {args.out}")
    if missing_src:
        print(f"WARNING: {len(missing_src)} source .xyz files not found in "
              f"{args.aligned_xyz}:")
        for sid in missing_src:
            print(f"  - {sid}")


if __name__ == "__main__":
    main()
