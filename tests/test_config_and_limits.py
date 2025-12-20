import tempfile
import yaml
from pathlib import Path
import sys

# Ensure project root is on sys.path so `scrape_pdb` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.config import load_config, PipelineConfig


def test_load_config_and_max_downloads(tmp_path):
    cfg = {
        "search_parameters": {"metal_ion": "ZN"},
        "processing": {
            "batch_size": 5,
            "temp_directory": str(tmp_path / "temp"),
            "max_downloads": 2
        },
        "output": {
            "results_database": str(tmp_path / "results" / "clusters.csv"),
            "checkpoint_file": str(tmp_path / "results" / "checkpoint.db"),
            "log_file": str(tmp_path / "results" / "pipeline.log"),
            "output_dir": str(tmp_path / "results")
        },
        "validation": {}
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    config = load_config(str(p))
    assert isinstance(config, PipelineConfig)
    assert config.processing.max_downloads == 2


# Note: Full integration tests (network, downloads, parsing) require
# setting up mocks for network calls and parser behavior. The above
# test validates parsing and availability of the new limit field.
