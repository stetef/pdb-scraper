#!/usr/bin/env python3
"""CLI ↔ library parity on the committed 5XP6 and 1OAO slices (card P1.9).

D-25 / 04 §2: the CLI (``parser.process_pdb``) stays a *parallel*
implementation of the cluster/altloc logic rather than becoming a thin caller of
``scrape_pdb.api`` — its output must stay byte-compatible. This test is what
replaces the refactor: for each committed slice, the sites the library reports
must be the same sites the CLI writes.

What "parity" covers here (04 §2, plus the card's coordinate/first-shell
requirement):

1. **Site set** — every xyz file the CLI writes corresponds to exactly one
   ``SiteCandidate`` with the same ``(cluster index, altloc label)`` and the same
   absorber atom (identified by its ``RES=/CHAIN=/RESSEQ=/ATOM=`` provenance tail
   and cross-checked against the PDB serial in ``target_atom_index``).
2. **Atom coordinates** — the absorber-centred heavy-atom set is identical
   (element, coordinates to 1e-3 Å, per-atom provenance tail). The CLI keeps
   selection order and uppercase symbols, the library puts the absorber first and
   title-cases, so the comparison is on sets, case-insensitively.
3. **First shell** — the coordination summary the library reports
   (element/count/rmin/rmax) is exactly what the coordination window picks out of
   the CLI's own xyz file.
4. **The one sanctioned difference** — a polynuclear cluster gives the library
   one candidate *per absorber atom* (04 §3.1), where the CLI writes a single
   file centred on the first target atom. Those extra candidates must be
   siblings of a matched one, never sites the CLI does not know about.

QUESTIONS Q-P1.9-2: 04 §2 says the set of ``(cluster_id, target serial,
altloc)`` from ``extract_sites`` *equals* the set of files ``process_pdb``
writes. Taken literally that contradicts 04 §3.1 (5XP6: one CLI file, two
candidates). Reading taken here, which the surrounding text favours: a
bijection CLI file -> candidate on ``(cluster, altloc, absorber)``, with the
extra per-absorber candidates required to be siblings of a matched one (point
4 above).

No network: both paths run on the local slices under ``tests/data``.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.api import ExtractConfig, extract_sites
from scrape_pdb.config import PipelineConfig
from scrape_pdb.constants import COORDINATION_DONOR_ELEMENTS
from scrape_pdb.parser import process_pdb

DATA = Path(__file__).resolve().parent / "data"

# One shared set of extraction settings for both paths, so any difference is a
# real divergence and not a configuration mismatch.
CUTOFF = 5.0            # metal-metal clustering distance (ExtractConfig default)
SELECTION_RADIUS = 6.0  # crop radius (ExtractConfig default)
COORD_MIN = 2.0
COORD_MAX = 2.8

FIXTURES = [
    ("5xp6_zn_site.pdb", "ZN"),   # di-zinc NDM-1 site: one cluster, two absorbers
    ("1oao_ni_site.pdb", "NI"),   # ACS/CODH Ni site next to an Fe4S4: three altlocs
]


def _cli_config(out_dir: Path, element: str) -> PipelineConfig:
    """The CLI configuration that matches ``ExtractConfig``'s defaults."""
    return PipelineConfig(
        **{
            "search_parameters": {"metal_ion": element},
            "processing": {
                "cutoff": CUTOFF,
                "selection_radius": SELECTION_RADIUS,
                "target": element,
                "temp_directory": str(out_dir / "downloads"),
                "include_waters": True,
            },
            "output": {
                "results_database": str(out_dir / "clusters.csv"),
                "checkpoint_file": str(out_dir / "checkpoint.db"),
                "log_file": str(out_dir / "pipeline.log"),
                "output_dir": str(out_dir),
            },
            "validation": {
                "coordination_distance_min": COORD_MIN,
                "coordination_distance_max": COORD_MAX,
            },
        }
    )


def _parse_xyz(text: str) -> tuple[dict[str, str], list[tuple]]:
    """Return (comment key=value fields, heavy atoms).

    An atom is ``(ELEMENT, x, y, z, provenance tail)``. Hydrogens are dropped:
    the CLI appends Open Babel hydrogens to its files (``add_hydrogens=True``),
    the library does not (``ExtractConfig.add_hydrogens=False``), and no H
    reaches FEFF anyway (04 §3.4).
    """
    lines = text.splitlines()
    n = int(lines[0].strip())
    fields = {}
    for part in lines[1].split():
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k] = v
    atoms = []
    for line in lines[2 : 2 + n]:
        body, _, tail = line.partition("#")
        p = body.split()
        if p[0].upper() == "H":
            continue
        atoms.append(
            (p[0].upper(), round(float(p[1]), 3), round(float(p[2]), 3), round(float(p[3]), 3), tail.strip())
        )
    return fields, atoms


def _absorber(atoms: list[tuple]) -> tuple:
    """The atom at the origin — both paths translate the absorber to (0,0,0)."""
    origin = [a for a in atoms if a[1] == 0.0 and a[2] == 0.0 and a[3] == 0.0]
    assert len(origin) == 1, f"expected exactly one atom at the origin, got {origin}"
    return origin[0]


def _first_shell(atoms: list[tuple]) -> dict[str, tuple[int, float, float]]:
    """Coordination summary recomputed from an xyz file, using the same window and
    donor-element rule as ``cluster.select_coordinating_neighbors``."""
    by_elem: dict[str, list[float]] = {}
    for elem, x, y, z, _tail in atoms:
        if elem not in COORDINATION_DONOR_ELEMENTS:
            continue
        d = math.sqrt(x * x + y * y + z * z)
        if COORD_MIN <= d <= COORD_MAX:
            by_elem.setdefault(elem, []).append(d)
    return {e: (len(v), round(min(v), 3), round(max(v), 3)) for e, v in by_elem.items()}


def _serial_index(pdb_path: Path) -> dict[int, str]:
    """PDB serial -> ``RES=… CHAIN=… RESSEQ=… ATOM=… REC=…`` tail, so the library's
    ``target_atom_index`` can be checked against the CLI's file content."""
    index = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        serial = int(line[6:11])
        resseq = f"{line[22:26].strip()}{line[26:27].strip()}".strip()
        index[serial] = (
            f"RES={line[17:20].strip()} CHAIN={line[21:22].strip()} "
            f"RESSEQ={resseq} ATOM={line[12:16].strip()} REC={line[:6].strip()}"
        )
    return index


@pytest.mark.parametrize("fixture,element", FIXTURES)
def test_cli_and_library_agree_on_sites(tmp_path, fixture, element):
    pdb_path = DATA / fixture
    serials = _serial_index(pdb_path)

    lib_work = tmp_path / "lib"
    lib_work.mkdir()
    cli_out = tmp_path / "cli"
    written, _stats = process_pdb(str(pdb_path), _cli_config(cli_out, element))
    assert written, "the CLI wrote no xyz file for this fixture"

    api = extract_sites(
        pdb_path,
        element=element,
        config=ExtractConfig(
            workdir=lib_work,
            cutoff=CUTOFF,
            selection_radius=SELECTION_RADIUS,
            coordination_distance_min=COORD_MIN,
            coordination_distance_max=COORD_MAX,
        ),
    )
    assert api.sites

    # The library never touches the input, and the CLI's own files live under its
    # own output dir — neither path writes into the other's workdir.
    assert not list(lib_work.iterdir())

    matched: list = []
    for path in written:
        fields, cli_atoms = _parse_xyz(Path(path).read_text())
        cluster_index = int(fields["CLUSTER"])
        altloc = fields.get("ALTLOC_LABEL")
        cli_absorber = _absorber(cli_atoms)

        # (1) exactly one candidate with this (cluster, altloc, absorber)
        candidates = [
            s
            for s in api.sites
            if s.cluster_id == f"cluster{cluster_index}"
            and s.altloc_label == altloc
            and serials[s.target_atom_index] == cli_absorber[4]
        ]
        assert len(candidates) == 1, (
            f"{Path(path).name}: expected one library site for cluster{cluster_index} "
            f"altloc={altloc} absorber={cli_absorber[4]!r}, got {len(candidates)}"
        )
        site = candidates[0]
        matched.append(site)

        assert site.altloc_case == (int(fields["ALTLOC_CASE"]) if "ALTLOC_CASE" in fields else None)
        assert site.cluster_type == fields["CLUSTER_TYPE"]
        assert site.pdb_id == fields["PDB"]

        # (2) same absorber-centred heavy-atom set
        _sfields, api_atoms = _parse_xyz(site.xyz_text)
        assert set(api_atoms) == set(cli_atoms), (
            f"{Path(path).name}: atom sets differ "
            f"(CLI-only {sorted(set(cli_atoms) - set(api_atoms))[:3]}, "
            f"library-only {sorted(set(api_atoms) - set(cli_atoms))[:3]})"
        )

        # (3) same first shell
        api_shell = {
            c["element"].upper(): (c["count"], c["rmin"], c["rmax"]) for c in site.coordination
        }
        assert api_shell == _first_shell(cli_atoms)
        assert site.geometry["CN"] == sum(c["count"] for c in site.coordination)

    # (4) the only extra library sites are the sanctioned per-absorber ones
    matched_serials = {s.target_atom_index for s in matched}
    for site in api.sites:
        if site.target_atom_index in matched_serials:
            continue
        assert any(
            site.target_atom_index in m.sibling_target_indices and site.cluster_id == m.cluster_id
            for m in matched
        ), (
            f"library reported a site (cluster {site.cluster_id}, serial "
            f"{site.target_atom_index}) that is not an absorber sibling of any CLI site"
        )


def test_5xp6_polynuclear_difference_is_the_expected_one(tmp_path):
    """The di-zinc cluster is where the two paths legitimately differ: one CLI
    file, two library candidates (04 §3.1)."""
    pdb_path = DATA / "5xp6_zn_site.pdb"
    written, _ = process_pdb(str(pdb_path), _cli_config(tmp_path / "cli", "ZN"))
    api = extract_sites(pdb_path, element="ZN", config=ExtractConfig(workdir=tmp_path / "lib", cutoff=CUTOFF))

    assert len(written) == 1
    assert len(api.sites) == 2
    assert {s.cluster_id for s in api.sites} == {"cluster1"}
    assert all(s.cluster_type == "multi_homo" for s in api.sites)
