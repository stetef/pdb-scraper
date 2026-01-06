import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.main import run_pipeline
from scrape_pdb.checkpoint import CheckpointManager


def test_pipeline_with_local_ids(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / 'tests' / 'data'
    ids_file = data_dir / 'ids.txt'
    pdb_downloads = data_dir / 'PDB-downloads'

    # Build a PipelineConfig-like YAML dict using list_file mode
    cfg = {
        'search_parameters': {
            'metal_ion': 'ZN',

            'resolution_cutoff': 3.0,
            'experimental_method': 'X-RAY DIFFRACTION',
            'polymer_type': 'Protein'
        },
        'processing': {
            'batch_size': 2,
            'parallel_workers': 1,
            'rate_limit_delay': 0.0,
            'temp_directory': str(tmp_path / 'downloads'),
            'cutoff': 6.0,
            'target': 'ZN',
            'metals_excluded': [],
            'max_downloads': 10,
            'must_have': 'S',
            'include_waters': True
        },
        'output': {
            'results_database': str(tmp_path / 'results' / 'clusters.csv'),
            'checkpoint_file': str(tmp_path / 'results' / 'checkpoint.db'),
            'log_file': str(tmp_path / 'results' / 'pipeline.log'),
            'save_matching_structures': False,
            'matched_structures_dir': str(tmp_path / 'results' / 'matched_cifs'),
            'output_dir': str(tmp_path / 'results')
        },
        'validation': {
            'coordination_distance_max': 2.8,
            'coordination_distance_min': 2.0
        },
        'input_mode': 'list_file',
        'input_data': [str(ids_file)],
        'log_level': 'INFO'
    }

    cfg_path = tmp_path / 'cfg_local.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Monkeypatch fetch_pdb to return local files from data/PDB-downloads
    def fake_fetch(pdb_id, dest_dir):
        pid = pdb_id.strip().lower()
        # try .pdb then .cif
        for ext in ('.pdb', '.cif'):
            candidate = pdb_downloads / f"{pid}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    monkeypatch.setattr('scrape_pdb.downloader.fetch_pdb', fake_fetch)

    # Run pipeline
    rc = run_pipeline(str(cfg_path), verbose=False)
    assert rc == 0

    # Verify checkpoint entries for IDs from ids.txt
    cp = CheckpointManager(str(cfg['output']['checkpoint_file']))
    # ids.txt contents like "9QQT,9QR3"
    raw = ids_file.read_text().strip()
    tokens = [t.strip().lower() for t in raw.replace(',', ' ').split() if t.strip()]
    for pid in tokens:
        status = cp.get_status(pid)
        # Each should have some status (matched/rejected) or None if not processed
        assert status in ('matched', 'rejected', 'error') or status is not None
