from pathlib import Path
import pytest


# Repository root
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    return repo_root / "data"


@pytest.fixture(scope="session")
def scripts_dir(repo_root: Path) -> Path:
    return repo_root / "scripts"
