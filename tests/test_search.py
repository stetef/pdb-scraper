import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_pdb import search
from scrape_pdb.config import (
    PipelineConfig,
    SearchParameters,
    ProcessingConfig,
    OutputConfig,
    ValidationConfig,
)


def make_config():
    sp = SearchParameters(
        metal_ion="ZN",

        resolution_cutoff=2.5,
        experimental_method="X-RAY DIFFRACTION",
        polymer_type="Protein",
    )
    return PipelineConfig(
        search_parameters=sp,
        processing=ProcessingConfig(),
        output=OutputConfig(),
        validation=ValidationConfig(),
    )


def test_search_pdb_success(monkeypatch):
    cfg = make_config()
    captured = {}

    def fake_post(url, json):
        captured['url'] = url
        captured['json'] = json

        class Resp:
            status_code = 200

            def json(self):
                return {'result_set': [{'identifier': '1ABC'}, {'identifier': '2DEF'}]}

        return Resp()

    monkeypatch.setattr('scrape_pdb.search.requests.post', fake_post)

    ids = search.search_pdb(cfg)
    assert ids == ['1ABC', '2DEF']

    nodes = captured['json']['query']['nodes']
    assert any(n['parameters'].get('value') == 'ZN' for n in nodes)


def test_search_pdb_failure(monkeypatch):
    cfg = make_config()

    def fake_post(url, json):
        class Resp:
            status_code = 500
            text = 'server error'

        return Resp()

    monkeypatch.setattr('scrape_pdb.search.requests.post', fake_post)

    ids = search.search_pdb(cfg)
    assert ids == []
