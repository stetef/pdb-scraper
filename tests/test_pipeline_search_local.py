import sys
import shutil
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.main import run_pipeline
from scrape_pdb.checkpoint import CheckpointManager


def make_search_config(tmp_path: Path) -> dict:
    return {
        'search_parameters': {
            'metal_ion': 'ZN',

            'resolution_cutoff': 3.0,
            'experimental_method': 'X-RAY DIFFRACTION',
            'polymer_type': 'Protein'
        },
        'processing': {
            'batch_size': 1,
            'parallel_workers': 1,
            'rate_limit_delay': 0.0,
            'temp_directory': str(tmp_path / 'downloads'),
            'cutoff': 6.0,
            'target': 'ZN',
            'metals_excluded': [],
            'max_downloads': 2,
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
        'input_mode': 'search',
        'input_data': [],
        'log_level': 'INFO'
    }


def test_search_mode_pipeline_uses_local_files_and_cleans(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / 'tests' / 'data'
    pdb_downloads = data_dir / 'PDB-downloads'

    cfg = make_search_config(tmp_path)
    cfg_path = tmp_path / 'cfg_search.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    # search should return list of IDs available in data/PDB-downloads
    available = [p.stem.upper() for p in pdb_downloads.iterdir() if p.suffix in ('.cif', '.pdb')]
    # enforce at least two available
    assert len(available) >= 2
    returned_ids = [available[0], available[1], *available[2:]]

    # Instead of letting resolve_input_sources download files into the download_dir
    # (which would be cleaned and could remove upcoming files), we monkeypatch
    # resolve_input_sources to place the real files into a separate staging dir
    # and also create dummy files in the configured download_dir so we can
    # verify the cleanup step runs and removes them.
    def fake_resolve(cfg_in, checkpoint=None):
        staging = tmp_path / 'staging'
        staging.mkdir(parents=True, exist_ok=True)
        paths = []
        maxd = cfg_in.processing.max_downloads or len(returned_ids)
        for pid in returned_ids[:maxd]:
            src = None
            for ext in ('.pdb', '.cif'):
                cand = pdb_downloads / f"{pid.lower()}{ext}"
                if cand.exists():
                    src = cand
                    break
            if src:
                dest = staging / src.name
                shutil.copy(src, dest)
                paths.append(str(dest))

        # create dummy files in the download_dir that should be removed by cleanup
        dld = Path(cfg_in.processing.temp_directory)
        dld.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            (dld / f"_tmp_cleanup_{i}").write_text('temp')

        return paths

    # Patch the reference used by `run_pipeline` which imports resolve_input_sources
    monkeypatch.setattr('scrape_pdb.main.resolve_input_sources', fake_resolve)

    rc = run_pipeline(str(cfg_path), verbose=False)
    assert rc == 0

    # After pipeline (batch_size=1), download dir should be empty because cleanup runs
    dl = Path(cfg['processing']['temp_directory'])
    assert not any(dl.iterdir())

    # Check checkpoint entries for the first two downloads (max_downloads=2)
    cp = CheckpointManager(str(cfg['output']['checkpoint_file']))
    for pid in returned_ids[:2]:
        status = cp.get_status(pid.lower())
        assert status in ('matched', 'rejected', 'error')

    # Ensure outputs were produced
    outdir = Path(cfg['output']['output_dir'])
    assert outdir.exists()
    clusters_path = Path(cfg['output']['results_database'])
    assert clusters_path.exists()
    # altloc report path default
    assert (outdir / 'altloc_report.csv').exists()
