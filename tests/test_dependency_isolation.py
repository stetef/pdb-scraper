#!/usr/bin/env python3
"""The library path must import on a core-only install (card P1.9, 04 §5, D-10).

EARL's API/web image installs ``scrape-pdb`` with its core dependencies only
(``biopython``, ``numpy``, ``pydantic``). Larch, matplotlib and 3D viewers must
never be pulled in at runtime (09 §3), and neither must the CLI's ``requests`` /
``tqdm`` / ``pyyaml``.

Two tests:

* :func:`test_import_api_pulls_no_cli_or_plot_dependencies` — cheap, always
  runs: imports ``scrape_pdb.api`` in a *fresh subprocess* (so modules another
  test already imported cannot mask a leak) and inspects ``sys.modules``.
* :func:`test_core_only_venv_can_import_api` — builds a throwaway venv with only
  the core dependencies. It needs ``uv`` and its package cache (i.e. possibly
  the network), so it is marked ``slow`` and opt-in via ``SCRAPE_PDB_TEST_VENV=1``
  — the default test run stays offline.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Top-level module names that must not be imported as a side effect of
# ``import scrape_pdb.api``. ``larch`` is the import name of ``xraylarch``.
FORBIDDEN_TOP_LEVEL = (
    "requests",
    "tqdm",
    "matplotlib",
    "larch",
    "py3dmol",
    "yaml",
    "openbabel",
)

# Run in a child interpreter: report every top-level module name that ended up
# in sys.modules after importing the library entry point.
_PROBE = """
import json, sys
import scrape_pdb.api  # noqa: F401
print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))
"""


def _import_probe(python: str, env_extra: dict | None = None) -> set[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("PYTHONSTARTUP", None)
    if env_extra:
        env.update(env_extra)
    out = subprocess.run(
        [python, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=300,
    )
    assert out.returncode == 0, f"import scrape_pdb.api failed:\n{out.stderr}"
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_import_api_pulls_no_cli_or_plot_dependencies():
    """``import scrape_pdb.api`` must not drag in any CLI/plot/DFT dependency."""
    loaded = _import_probe(sys.executable)
    leaked = sorted(loaded & set(FORBIDDEN_TOP_LEVEL))
    assert not leaked, (
        f"importing scrape_pdb.api pulled in {leaked}; keep those imports lazy "
        "or out of the library path (04 §5, D-10)"
    )
    # Sanity: the probe really did import the package.
    assert "scrape_pdb" in loaded


def test_import_package_root_pulls_no_cli_dependencies():
    """The same holds for the package root — ``scrape_pdb/__init__`` is on the path
    of every ``import scrape_pdb.api``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run(
        [sys.executable, "-c",
         "import json, sys; import scrape_pdb; "
         "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert out.returncode == 0, out.stderr
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert not sorted(loaded & set(FORBIDDEN_TOP_LEVEL))


def test_run_pipeline_is_still_reachable_from_the_package_root():
    """Making the CLI import lazy must not remove it from the public surface."""
    import scrape_pdb

    assert callable(scrape_pdb.run_pipeline)
    assert "run_pipeline" in dir(scrape_pdb)
    with pytest.raises(AttributeError):
        scrape_pdb.no_such_attribute


@pytest.mark.slow
def test_core_only_venv_can_import_api(tmp_path):
    """Install the project with *core deps only* into a throwaway venv and import
    the library API there.

    Opt-in (``SCRAPE_PDB_TEST_VENV=1``) because building the venv may need to
    reach the package index; the rest of the suite is offline.
    """
    if not os.environ.get("SCRAPE_PDB_TEST_VENV"):
        pytest.skip("set SCRAPE_PDB_TEST_VENV=1 to build a throwaway venv (needs uv + its cache)")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available")

    venv = tmp_path / "core-only"
    subprocess.run([uv, "venv", str(venv)], check=True, capture_output=True, timeout=600)
    python = venv / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-cache-dir", str(ROOT)],
        check=True, capture_output=True, timeout=1800,
    )

    # Import from the installed package, not the source tree.
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    out = subprocess.run(
        [str(python), "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
    )
    assert out.returncode == 0, f"core-only import failed:\n{out.stderr}"
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert not sorted(loaded & set(FORBIDDEN_TOP_LEVEL))

    # Belt and braces: the forbidden distributions are not even installed.
    listing = subprocess.run(
        [uv, "pip", "list", "--python", str(python)],
        capture_output=True, text=True, check=True, timeout=600,
    ).stdout.lower()
    for dist in ("xraylarch", "matplotlib", "py3dmol", "requests", "tqdm"):
        assert f"\n{dist} " not in f"\n{listing}", f"{dist} must not be a core dependency"
