import gzip
import io
import json

import pandas as pd

import lipinet.databases as db
from lipinet.databases import download_and_load_data


class DummyResp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad status")


def test_download_csv_then_load_no_redownload(monkeypatch):
    csv_text = "a,b\n1,2\n"
    calls = {"count": 0}

    def fake_get(_url):
        calls["count"] += 1
        return DummyResp(csv_text.encode("utf-8"))

    monkeypatch.setattr(db, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    # Force first download and then use local
    df1 = download_and_load_data(
        "unit_test_io_first.csv",
        url="http://example/csv",
        file_format="csv",
        force_download=True,
    )
    assert isinstance(df1, pd.DataFrame)
    df2 = download_and_load_data(
        "unit_test_io_first.csv",
        url="http://example/csv",
        file_format="csv",
    )
    assert isinstance(df2, pd.DataFrame)
    assert calls["count"] == 1


def test_download_gzip_tsv(monkeypatch):
    tsv_text = "x\ty\nA\tB\n"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(tsv_text.encode("latin-1"))
    content = buf.getvalue()

    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(content))}),
    )

    df = download_and_load_data(
        "unit_test_io.tsv",
        url="http://example/tsv",
        file_format="tsv",
        compressed=True,
        sep="\t",
        encoding="latin-1",
        force_download=True,
    )
    assert list(df.columns) == ["x", "y"]


def test_download_json(monkeypatch):
    payload = {"a": 1, "b": [1, 2]}
    content = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(content))}),
    )

    out = download_and_load_data(
        "unit_test.json",
        url="http://example/json",
        file_format="json",
        force_download=True,
    )
    assert isinstance(out, dict)
    assert out == payload


def test_get_prior_knowledge_swisslipids_gz(monkeypatch):
    # Prepare a tiny SwissLipids TSV with key columns
    tsv = (
        "Lipid ID\tLipid class*\tCHEBI\tComponents*\n"
        "L1\t Class A  \t CHEBI:123 \tFA(16:0)/FA(18:1)\n"
    )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(tsv.encode("latin-1"))
    content = buf.getvalue()

    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(content))}),
    )

    df = db.get_prior_knowledge("swisslipids", verbose=True, force_download=True)
    assert isinstance(df, pd.DataFrame)
    # Cleaned: strip spaces and remove CHEBI: prefix
    assert df.loc[0, "Lipid class*"] == "Class A"
    assert df.loc[0, "CHEBI"] == "123"


def test_get_prior_knowledge_rhea_tsv(monkeypatch):
    # Minimal Rhea TSV with headers used by parse_rhea
    tsv = (
        "Reaction identifier\tEquation\tChEBI identifier\tChEBI name\tEC number\tEnzymes\tGene Ontology\tCross-reference (Reactome)\n"
        "R1\tA=B\tCHEBI:1;CHEBI:2\tA;B\tEC:1.2.3.4;EC:5.6.7.8\tE1\tGO:1\tRXN:1\n"
    )
    content = tsv.encode("utf-8")
    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(content))}),
    )
    df = db.get_prior_knowledge("rhea", force_download=True)
    assert isinstance(df, pd.DataFrame)
    assert "Reaction identifier" in df.columns


def test_get_prior_knowledge_unsupported_resource():
    try:
        db.get_prior_knowledge("unknown_resource")
    except KeyError as e:
        assert "not yet supported" in str(e)
    else:
        raise AssertionError("Expected KeyError for unsupported resource")


def test_download_and_load_data_verbose_and_errors(monkeypatch, _tmp_path=None):
    # Exercise verbose branches and error paths
    # Unsupported file format
    # Ensure no network by mocking requests.get
    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(b""))}),
    )
    try:
        download_and_load_data(
            "x.bin",
            url="http://example/x",
            file_format="bin",
            verbose=True,
            force_download=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsupported format")

    # Compressed with non-csv/tsv
    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(b"junk"))}),
    )
    try:
        download_and_load_data(
            "x.json.gz",
            url="http://example/x",
            file_format="json",
            compressed=True,
            verbose=True,
            force_download=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for gzip+json unsupported path")

    # Verbose path when file exists locally (CSV)
    csv_text = "a,b\n1,2\n"
    monkeypatch.setattr(
        db,
        "requests",
        type("R", (), {"get": staticmethod(lambda _u: DummyResp(csv_text.encode("utf-8")))}),
    )
    _ = download_and_load_data(
        "exists.csv",
        url="http://example/x",
        file_format="csv",
        verbose=True,
        force_download=True,
    )
    _ = download_and_load_data(
        "exists.csv",
        url="http://example/x",
        file_format="csv",
        verbose=True,
    )
