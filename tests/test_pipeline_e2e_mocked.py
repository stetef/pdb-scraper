import sys
import yaml
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.main import run_pipeline
from scrape_pdb.checkpoint import CheckpointManager
from scrape_pdb.config import PipelineConfig, SearchParameters, ProcessingConfig, OutputConfig, ValidationConfig


def make_config_dict(tmp_path: Path) -> dict:
    return {
        'search_parameters': {
            'metal_ion': 'ZN',
            'coordinating_residues': ['CYS'],
            'coordination_count': 4,
            'resolution_cutoff': 2.5,
            'experimental_method': 'X-RAY DIFFRACTION',
            'polymer_type': 'Protein',
        },
        'processing': {
            'batch_size': 3,
            'parallel_workers': 1,
            'rate_limit_delay': 0.0,
            'temp_directory': str(tmp_path / 'downloads'),
            'cutoff': 3.0,
            'target': 'ZN',
            'metals_excluded': [],
            'max_downloads': 10,
            'must_have': '',
            'include_waters': True,
        },
        'output': {
            'results_database': str(tmp_path / 'results' / 'clusters.csv'),
            'checkpoint_file': str(tmp_path / 'results' / 'checkpoint.db'),
            'log_file': str(tmp_path / 'results' / 'pipeline.log'),
            'save_matching_structures': False,
            'matched_structures_dir': str(tmp_path / 'results' / 'matched_cifs'),
            'output_dir': str(tmp_path / 'results'),
        },
        'validation': {
            'coordination_distance_max': 2.8,
            'coordination_distance_min': 2.0,
        },
        'input_mode': 'search',
        'input_data': [],
        'log_level': 'DEBUG',
    }


def test_pipeline_end_to_end_mocked(monkeypatch, tmp_path):
    cfg_dict = make_config_dict(tmp_path)
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg_dict))

    # Prepare 10 fake pdb ids and files in download dir
    download_dir = Path(cfg_dict['processing']['temp_directory'])
    download_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = [f"P{i:03d}" for i in range(10)]
    files = []
    for pid in pdb_ids:
        p = download_dir / f"{pid.lower()}.pdb"
        p.write_text(f"REMARK   2 RESOLUTION.    1.80 ANGSTROM.\nATOM\n")
        files.append(str(p))

    # Monkeypatch resolve_input_sources to return the files (simulate search+download)
    def fake_resolve(config, checkpoint=None):
        # return first N based on max_downloads
        maxd = config.processing.max_downloads
        return files[: maxd if maxd is not None else len(files)]

    monkeypatch.setattr('scrape_pdb.main.resolve_input_sources', fake_resolve)

    # Create behavior map for process_pdb: some matched (return path), some rejected (empty), some errors
    behavior = {}
    # first 4 matched, next 4 rejected, last 2 raise
    for i, pid in enumerate(pdb_ids):
        if i < 4:
            behavior[pid.lower()] = ('matched', [f"/fake/{pid}_1.xyz"]) 
        elif i < 8:
            behavior[pid.lower()] = ('rejected', [])
        else:
            behavior[pid.lower()] = ('error', RuntimeError('simulated'))

    def fake_process_pdb(pdb_path: str, config):
        pid = Path(pdb_path).stem
        kind = behavior[pid][0]
        if kind == 'matched':
            return behavior[pid][1]
        elif kind == 'rejected':
            return []
        else:
            raise behavior[pid][1]

    monkeypatch.setattr('scrape_pdb.main.process_pdb', fake_process_pdb)

    # Run pipeline
    rc = run_pipeline(str(cfg_path), verbose=False)

    # rc should be non-zero because of errors
    assert rc != 0

    # Verify checkpoint statuses
    cp = CheckpointManager(str(cfg_dict['output']['checkpoint_file']))
    for i, pid in enumerate(pdb_ids):
        p = pid.lower()
        status = cp.get_status(p)
        if i < 4:
            assert status == 'matched'
        elif i < 8:
            assert status == 'rejected'
        else:
            assert status == 'error'

    # Verify download dir cleaned (cleanup after batches of 3 should have removed files)
    assert not any(download_dir.iterdir())


def test_pipeline_search_parameters_cause_rejection(monkeypatch, tmp_path):
    # Build a config where search parameters are strict
    cfg_dict = make_config_dict(tmp_path)
    # tighten parameters
    cfg_dict['search_parameters']['resolution_cutoff'] = 1.0
    cfg_dict['search_parameters']['experimental_method'] = 'NMR'
    cfg_dict['search_parameters']['polymer_type'] = 'DNA'
    cfg_path = tmp_path / 'cfg2.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg_dict))

    # Prepare fake IDs that search will return
    fake_ids = [f"X{i:02d}" for i in range(5)]

    captured = {}

    def fake_search(cfg):
        # capture that all search params were passed through
        captured['params'] = cfg.search_parameters
        return fake_ids

    # downloader.resolve_input_sources imports search_pdb from scrape_pdb.search
    monkeypatch.setattr('scrape_pdb.downloader.search_pdb', fake_search)

    # Monkeypatch fetch_pdb to write simple PDB files and return paths
    dl_dir = Path(cfg_dict['processing']['temp_directory'])
    dl_dir.mkdir(parents=True, exist_ok=True)

    def fake_fetch(pdb_id, dest_dir):
        p = Path(dest_dir) / f"{pdb_id.lower()}.pdb"
        p.write_text("REMARK   2 RESOLUTION.    3.00 ANGSTROM.\nATOM\n")
        return str(p)

    monkeypatch.setattr('scrape_pdb.downloader.fetch_pdb', fake_fetch)

    # Force process_pdb to always return [] so pipeline marks them rejected
    def fake_process(pdb_path, config):
        return []

    monkeypatch.setattr('scrape_pdb.main.process_pdb', fake_process)

    # Run pipeline
    rc = run_pipeline(str(cfg_path), verbose=False)
    # Should be success (no exceptions), but all entries rejected => rc == 0
    assert rc == 0

    # Check that search saw the strict parameters
    sp = captured.get('params')
    assert sp is not None
    assert sp.resolution_cutoff == 1.0
    assert sp.experimental_method.upper() == 'NMR'
    assert sp.polymer_type == 'DNA'

    # Verify checkpoint statuses are 'rejected'
    cp = CheckpointManager(str(cfg_dict['output']['checkpoint_file']))
    for pid in fake_ids:
        status = cp.get_status(pid.lower())
        assert status == 'rejected'

