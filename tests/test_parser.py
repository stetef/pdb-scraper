import sys
from pathlib import Path
import textwrap

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb import parser
from scrape_pdb.models import Atom


def test_detect_file_format(tmp_path):
    p_pdb = tmp_path / "1.pdb"
    p_pdb.write_text("ATOM      1\n")
    assert parser.detect_file_format(str(p_pdb)) == "pdb"

    p_cif = tmp_path / "1.cif"
    p_cif.write_text("data_test\n")
    assert parser.detect_file_format(str(p_cif)) == "cif"


def test_load_pdb_atoms_and_collapse_and_resolution(tmp_path):
    # Build properly formatted PDB lines so fixed-width parsing works
    def make_line(rec, serial, name, altloc, resname, chain, resseq, x, y, z, element):
        # PDB fixed-width columns approximated for parser slicing
        return f"{rec:<6}{serial:5d} {name:<4}{altloc:1}{resname:>3} {chain:1}{resseq:4}{' ':1}   {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{10.00:6.2f}          {element:>2}\n"

    lines = []
    lines.append("REMARK   2 RESOLUTION.    1.80 ANGSTROM.\n")
    lines.append(make_line('ATOM', 1, 'N', '', 'ALA', 'A', 1, 11.104, 13.207, 2.0, 'N'))
    lines.append(make_line('HETATM', 2, 'ZN', '', 'ZN', 'B', 2, 0.0, 0.0, 0.0, 'ZN'))
    # Two SG altlocs for same atom name/resseq; one altloc A and one B
    lines.append(make_line('HETATM', 3, 'SG', 'A', 'CYS', 'A', 3, 1.0, 0.0, 0.0, 'S'))
    lines.append(make_line('HETATM', 4, 'SG', 'B', 'CYS', 'A', 3, 1.0, 0.0, 0.0, 'S'))
    lines.append('END\n')

    p = tmp_path / "t.pdb"
    p.write_text(''.join(lines))

    atoms = parser.load_pdb_atoms_all(str(p))
    # should parse 4 atom lines
    assert len(atoms) == 4

    # collapse altloc should pick blank/A over B when policy prefer_blank_or_A
    collapsed = parser.collapse_altloc(atoms, policy="prefer_blank_or_A")
    # should include SG after collapsing duplicates
    names = [a.atom_name for a in collapsed]
    assert any(n.strip() == "SG" for n in names)

    res = parser.parse_pdb_resolution(str(p))
    assert isinstance(res, float) and abs(res - 1.8) < 1e-6


def test_iter_altloc_metal_records(tmp_path):
    # Use same formatting helper as above
    def make_line(rec, serial, name, altloc, resname, chain, resseq, x, y, z, element):
        return f"{rec:<6}{serial:5d} {name:<4}{altloc:1}{resname:>3} {chain:1}{resseq:4}{' ':1}   {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{10.00:6.2f}          {element:>2}\n"

    lines = []
    # first ZN has blank altloc -> ignored by iterator
    lines.append(make_line('HETATM', 1, 'ZN', '', 'ZN', 'A', 1, 0.0, 0.0, 0.0, 'ZN'))
    # second ZN has altloc 'B' -> should be yielded
    lines.append(make_line('HETATM', 2, 'ZN', 'B', 'ZN', 'A', 1, 1.0, 0.0, 0.0, 'ZN'))
    # SG with altloc B but not a metal -> ignored
    lines.append(make_line('HETATM', 3, 'SG', 'B', 'SG', 'A', 2, 1.0, 1.0, 0.0, 'S'))

    p = tmp_path / "alt.pdb"
    p.write_text(''.join(lines))

    metals = set(["ZN", "FE"])  # look for ZN
    recs = list(parser.iter_altloc_metal_records(str(p), metals))
    assert any(r.element.upper() == "ZN" for r in recs)


def test_load_cif_atoms_all_monkeypatch(monkeypatch, tmp_path):
    # Monkeypatch MMCIF2Dict to return a dict-like object
    fake = {
        '_atom_site.group_PDB': ['HETATM', 'ATOM'],
        '_atom_site.id': ['1', '2'],
        '_atom_site.label_atom_id': ['ZN','N'],
        '_atom_site.label_alt_id': ['A',''],
        '_atom_site.label_comp_id': ['ZN','ALA'],
        '_atom_site.label_asym_id': ['A','A'],
        '_atom_site.label_seq_id': ['1','1'],
        '_atom_site.pdbx_PDB_ins_code': ['',''],
        '_atom_site.Cartn_x': ['0.0','1.0'],
        '_atom_site.Cartn_y': ['0.0','1.0'],
        '_atom_site.Cartn_z': ['0.0','1.0'],
        '_atom_site.occupancy': ['1.0','1.0'],
        '_atom_site.B_iso_or_equiv': ['0.0','0.0'],
        '_atom_site.type_symbol': ['ZN','N'],
        '_atom_site.pdbx_formal_charge': ['',''],
    }

    class FakeMM:
        def __init__(self, path):
            self._d = fake
        def get(self, key, default=None):
            return self._d.get(key, default)

    monkeypatch.setattr(parser, 'MMCIF2Dict', FakeMM)

    # write a dummy cif file
    p = tmp_path / 'f.cif'
    p.write_text('data_')
    atoms = parser.load_cif_atoms_all(str(p))
    assert isinstance(atoms, list)
    assert any(a.element.upper() in ("ZN","N") for a in atoms)


def test_extract_pdb_model_blocks_handles_unlabeled_model_records(tmp_path):
    pdb = tmp_path / "models.pdb"
    pdb.write_text(
        "REMARK   2 RESOLUTION.    2.00 ANGSTROM.\n"
        "MODEL        1\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N\n"
        "ENDMDL\n"
        "MODEL\n"
        "ATOM      2  N   GLY A   1       1.000   0.000   0.000  1.00 10.00           N\n"
        "ENDMDL\n"
        "END\n"
    )

    header, blocks = parser._extract_pdb_model_blocks(str(pdb))
    assert len(blocks) == 2
    assert any("RESOLUTION" in line for line in header)
    assert "ATOM" in blocks[0][0]
    assert "ATOM" in blocks[1][0]


def test_process_pdb_multi_model_adds_conformer_suffix(monkeypatch, tmp_path):
    pdb = tmp_path / "2lce.pdb"
    pdb.write_text(
        "REMARK   2 RESOLUTION.    2.00 ANGSTROM.\n"
        "MODEL        1\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      2  N   GLY A   1       1.000   0.000   0.000  1.00 10.00           N\n"
        "ENDMDL\n"
        "END\n"
    )

    seen_stems: list[str] = []

    def fake_single(pdb_path, config):
        stem = Path(pdb_path).stem
        seen_stems.append(stem)
        return [f"/tmp/{stem}_cluster1.xyz"], {
            "pdb_id": stem,
            "rejections": {"must_have": 1},
            "ligand_requirements": None,
            "rejected_cn_distribution": {4: 1},
            "rejected_coord_distribution": {"2N2S": 1},
        }

    monkeypatch.setattr(parser, "_process_single_pdb", fake_single)

    class DummyConfig:
        pass

    written, stats = parser.process_pdb(str(pdb), DummyConfig())

    assert seen_stems == ["2lce_conformer1", "2lce_conformer2"]
    assert all("_conformer" in p for p in written)
    assert stats["pdb_id"] == "2lce"
    assert stats["rejections"]["must_have"] == 2


def test_format_target_suffix_for_xyz_name():
    assert parser._format_target_suffix("ZN") == "Zn"
    assert parser._format_target_suffix("ni") == "Ni"
    assert parser._format_target_suffix("C") == "C"
