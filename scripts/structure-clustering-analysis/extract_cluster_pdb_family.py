"""
Extract cluster, PDB ID, and family columns from kmeans_labels_with_stats.csv.
Writes a sister CSV sorted by cluster (asc), then PDB ID (asc).

Usage:
    python extract_cluster_pdb_family.py <path/to/kmeans_labels_with_stats.csv>
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to kmeans_labels_with_stats.csv")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)

    df["pdb_id"] = df["id"].str[:4]

    result = (
        df[["cluster", "pdb_id", "family"]]
        .sort_values(["cluster", "pdb_id"])
        .reset_index(drop=True)
    )

    output_path = input_path.parent / "cluster_pdb_family.csv"
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
