# Examples

This folder contains ready-to-run configs for the pipeline.

- [config.search.yaml](config.search.yaml): Search-mode example using the RCSB Search API.
  - Adjust `search_parameters` (metal, resolution, method, polymer type) for your query.
  - Tune `processing` (batch_size, temp_directory, cutoff, target, max_downloads) for throughput and disk use.
  - Set `output` paths for results, checkpoint, logs, and matched structures (planned).
  - Update `validation` thresholds and `coord` (e.g., `4S`, `1N3S`) to keep only matching coordination environments.

Run the example:
```bash
uv run python -m scrape_pdb examples/config.search.yaml --verbose
```
