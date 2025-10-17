from pathlib import Path

import pandas as pd

import lipinet.utils as U


def test_clean_columns_uppercase_and_whitespace():
    df = pd.DataFrame({"x": ["  a  b   c "]})
    out = U.clean_columns(df, cols=["x"], uppercase=True, collapse_whitespace=True)
    assert out.loc[0, "x"] == "A B C"


def test_cache_paths_helper(tmp_path, monkeypatch):
    # Point cache root away from repo for isolation
    monkeypatch.setattr(U, "CACHE_ROOT", tmp_path, raising=True)
    p = U._cache_paths("sourceX")
    assert isinstance(p["nodes"], Path) and str(p["nodes"]).endswith("sourceX_nodes.pkl")
    assert isinstance(p["edges"], Path) and str(p["edges"]).endswith("sourceX_edges.pkl")

