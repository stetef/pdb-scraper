import sys
from pathlib import Path
import urllib.request
from types import SimpleNamespace

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb.downloader import fetch_pdb, resolve_input_sources, batch_download_from_list


class FakeResp:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status
    def read(self):
        return self._data
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_pdb_downloads_and_returns_path(tmp_path, monkeypatch):
    data = b"HEADER TEST\nATOM 1\nEND\n"
    def fake_urlopen(req, timeout=30):
        return FakeResp(data)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = fetch_pdb("1abc", str(tmp_path))
    assert out is not None
    p = Path(out)
    assert p.exists()
    assert p.read_bytes() == data


def test_batch_download_from_list_and_max_downloads(tmp_path, monkeypatch):
    data = b"HEADER\nATOM 1\nEND\n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: FakeResp(data))

    list_file = tmp_path / "ids.txt"
    list_file.write_text("1abc\n2def\n3ghi")

    paths = batch_download_from_list(str(list_file), str(tmp_path))
    assert len(paths) == 3

    # Now test resolve_input_sources with max_downloads limiting
    cfg = SimpleNamespace(
        input_mode="list_file",
        input_data=[str(list_file)],
        download_dir=tmp_path,
        processing=SimpleNamespace(max_downloads=2),
    )

    sources = resolve_input_sources(cfg)
    assert len(sources) == 2


def test_resolve_input_ids_and_checkpoint_skip(tmp_path, monkeypatch):
    # mock network
    data = b"PDBDATA\n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: FakeResp(data))

    # config with ids
    cfg = SimpleNamespace(
        input_mode="ids",
        input_data=["1abc", "2def"],
        download_dir=tmp_path,
        processing=SimpleNamespace(max_downloads=None),
    )

    class FakeCP:
        def get_status(self, pdb_id):
            if pdb_id == "1abc":
                return "matched"
            return None

    sources = resolve_input_sources(cfg, checkpoint=FakeCP())
    # should skip 1abc, only download 2def
    assert len(sources) == 1
    assert sources[0].endswith("2def.pdb") or sources[0].endswith("2def.cif")


def test_search_mode_uses_search_and_downloads(tmp_path, monkeypatch):
    # patch search_pdb to return ids
    monkeypatch.setattr("scrape_pdb.downloader.search_pdb", lambda config: ["1abc", "2def"]) 
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: FakeResp(b"X"))

    cfg = SimpleNamespace(
        input_mode="search",
        input_data=[],
        download_dir=tmp_path,
        processing=SimpleNamespace(max_downloads=None),
    )

    sources = resolve_input_sources(cfg)
    assert len(sources) == 2


def test_mixed_mode_paths_and_ids(tmp_path, monkeypatch):
    # create a local file
    local = tmp_path / "local.pdb"
    local.write_text("ATOM 1\n")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: FakeResp(b"P"))

    cfg = SimpleNamespace(
        input_mode="mixed",
        input_data=[str(local), "1abc", "bad_item"],
        download_dir=tmp_path,
        processing=SimpleNamespace(max_downloads=None),
    )

    sources = resolve_input_sources(cfg)
    # should include local file and downloaded 1abc
    assert any(str(local) == s for s in sources)
    assert any(s.endswith("1abc.pdb") or s.endswith("1abc.cif") for s in sources)
