from pathlib import Path

# Lightweight module to expose repository resource paths for tests
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "tests" / "data"
SCRIPTS_DIR = ROOT / "tests" / "scripts"

__all__ = ["DATA_DIR", "SCRIPTS_DIR"]
