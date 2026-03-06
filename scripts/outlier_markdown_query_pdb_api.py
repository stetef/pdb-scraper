#!/usr/bin/env python3
"""
Fetches PDB metadata (title, classification, organism, deposit date)
for every PDB ID found in the outlier summary markdown file,
then writes an annotated version.
"""

import re
import json
import argparse
import urllib.request
import urllib.error

DEFAULT_INPUT_FILE = "/mnt/user-data/uploads/zn_cluster_outlier_summary.md"
DEFAULT_OUTPUT_FILE = "/mnt/user-data/outputs/zn_cluster_outlier_summary_annotated.md"

# ── 2. Fetch metadata from RCSB GraphQL endpoint ───────────────────────────────
GRAPHQL_URL = "https://data.rcsb.org/graphql"

def fetch_metadata(pdb_id: str) -> dict:
    query = """
    {
      entry(entry_id: "%s") {
        struct { title }
        struct_keywords { pdbx_keywords }
                rcsb_accession_info { deposit_date }
        polymer_entities {
          rcsb_entity_source_organism { ncbi_scientific_name }
        }
      }
    }
    """ % pdb_id.upper()

    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response = json.loads(resp.read())
    except Exception as e:
        print(f"  [WARN] {pdb_id}: {e}")
        return {}

    if response.get("errors"):
        first_error = response["errors"][0].get("message", "unknown GraphQL error")
        print(f"  [WARN] {pdb_id}: GraphQL error: {first_error}")
        return {}

    data = (response.get("data") or {}).get("entry")

    if data is None:
        return {}

    title          = (data.get("struct") or {}).get("title", "N/A")
    classification = (data.get("struct_keywords") or {}).get("pdbx_keywords", "N/A")
    deposit_date   = (data.get("rcsb_accession_info") or {}).get("deposit_date", "N/A")

    # Collect unique organism names across all polymer entities
    organisms = []
    for entity in (data.get("polymer_entities") or []):
        for org in (entity.get("rcsb_entity_source_organism") or []):
            name = (org or {}).get("ncbi_scientific_name")
            if name and name not in organisms:
                organisms.append(name)
    organism = "; ".join(organisms) if organisms else "N/A"

    return {
        "title":          title.strip(),
        "classification": classification.strip() if classification else "N/A",
        "organism":       organism,
        "deposit_date":   deposit_date[:10] if deposit_date else "N/A",   # YYYY-MM-DD
    }

# ── 3. Rewrite the markdown, appending metadata after each filename ────────────
def format_meta(m: dict) -> str:
    if not m:
        return " — *metadata unavailable*"

    title = (m.get("title") or "N/A").replace("|", "\\|")
    classification = (m.get("classification") or "N/A").replace("|", "\\|")
    organism = (m.get("organism") or "N/A").replace("|", "\\|")
    deposit_date = (m.get("deposit_date") or "N/A").replace("|", "\\|")

    return (
        f" — **{title}**; "
        f"Classification: {classification}; "
        f"Organism: {organism}; "
        f"Deposited: {deposit_date}"
    )

def build_replacer(meta_lookup: dict[str, dict]):
    def replace_filename(match):
        full    = match.group(0)          # e.g. `2h59_ZN_homo_d2.60_cluster2.xyz`
        pdb_id  = match.group(2).lower()  # e.g. 2h59
        m       = meta_lookup.get(pdb_id, {})
        return full + format_meta(m)

    return replace_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PDB metadata for IDs found in an outlier summary markdown and "
            "write an annotated markdown file."
        )
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to input markdown file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to output markdown file (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Extract all unique PDB IDs ──────────────────────────────────────────
    with open(args.input_file) as f:
        content = f.read()

    # Filenames look like  `XXXX_ZN_homo_...`  where XXXX is the 4-char PDB id
    pdb_ids = sorted(set(re.findall(r'`([a-z0-9]{4})_ZN_homo', content, re.IGNORECASE)))
    print(f"Found {len(pdb_ids)} unique PDB IDs: {pdb_ids}")

    meta: dict[str, dict] = {}
    for pid in pdb_ids:
        print(f"  Fetching {pid.upper()} …")
        meta[pid.lower()] = fetch_metadata(pid)

    # Match backtick-wrapped filenames that contain a PDB id
    pattern = r'(`([a-z0-9]{4})_ZN_homo[^`]*`)'
    annotated = re.sub(pattern, build_replacer(meta), content, flags=re.IGNORECASE)

    # Add a note at the top
    header_note = (
        "> **Annotated version** — PDB metadata (title, classification, organism, "
        "deposit date) fetched from RCSB PDB API.\n\n"
    )
    annotated = header_note + annotated

    with open(args.output_file, "w") as f:
        f.write(annotated)

    print(f"\nDone! Annotated file written to:\n  {args.output_file}")


if __name__ == "__main__":
    main()