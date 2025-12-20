#!/usr/bin/env python3
import shutil
from pathlib import Path
import sys

repo = Path(__file__).resolve().parent.parent
data_output = repo / 'tests' / 'data' / 'output'
backup = repo / 'tests' / 'data' / 'output_backup_for_test'
if backup.exists():
    shutil.rmtree(backup)
if data_output.exists():
    shutil.copytree(data_output, backup)

# Monkeypatch fetch_pdb to prefer local files
import sys
sys.path.insert(0, str(repo))
import types
import sys as _sys

# Provide a lightweight fake Bio.MMCIF2Dict if Bio not installed so parser imports succeed
fake_bio = types.ModuleType('Bio')
fake_pdb = types.ModuleType('Bio.PDB')
fake_mmcif_mod = types.ModuleType('Bio.PDB.MMCIF2Dict')

def MMCIF2Dict_stub(path):
    # Return minimal dict-like mapping expected by parser when invoked
    return {}

fake_mmcif_mod.MMCIF2Dict = MMCIF2Dict_stub
_sys.modules['Bio'] = fake_bio
_sys.modules['Bio.PDB'] = fake_pdb
_sys.modules['Bio.PDB.MMCIF2Dict'] = fake_mmcif_mod

from scrape_pdb import main
import scrape_pdb.downloader as dl
real_fetch = dl.fetch_pdb

from pathlib import Path

def fake_fetch(pdb_id, dest_dir):
    pid = pdb_id.strip().lower()
    for ext in ('.pdb', '.cif'):
        candidate = Path(dest_dir) / f"{pid}{ext}"
        if candidate.exists():
            print(f"Using local file for {pdb_id}: {candidate}")
            return str(candidate)
    return real_fetch(pdb_id, dest_dir)


dl.fetch_pdb = fake_fetch

from difflib import unified_diff

def cmp_file(a, b):
    if not a.exists() or not b.exists():
        return False, 'missing file'
    a_lines = a.read_text().splitlines(keepends=True)
    b_lines = b.read_text().splitlines(keepends=True)
    if a_lines == b_lines:
        return True, ''
    diff = ''.join(unified_diff(a_lines, b_lines, fromfile=str(a), tofile=str(b)))
    return False, diff


def test_run_and_compare_local():
    rc = main.run_pipeline(str(repo / 'tests' / 'data' / 'config.json'), verbose=False)
    print('Pipeline returned rc=', rc)

    files_to_check = ['clusters_summary.csv', 'altloc_report.csv']
    all_same = True
    diffs = []
    for fname in files_to_check:
        before = backup / fname
        after = data_output / fname
        same, info = cmp_file(before, after)
        print(f"Comparing {fname}:", 'SAME' if same else 'DIFFERENT')
        if not same:
            all_same = False
            diffs.append((fname, info))

    if all_same:
        print('All checked files are identical to backup')
    else:
        for fname, info in diffs:
            print(f"Diff for {fname}:\n{info}")
    assert all_same, 'Output files differ from expected backup'