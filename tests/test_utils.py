from pathlib import Path

import numpy as np
import pandas as pd

import lipinet.utils as U
from lipinet.utils import (
    check_for_split_characters,
    clean_columns,
    clean_missing_strings,
    create_nodedf_from_edgedf,
    split_and_expand_large,
)


def test_split_and_expand_large_mixed_values():
    df = pd.DataFrame(
        {
            "col1": ["a|b", None, "c", np.nan, "d|e|f", ""],
            "keep": [1, 2, 3, 4, 5, 6],
        },
    )
    out = split_and_expand_large(df, split_col="col1", delimiter="|", expand_cols=["keep"])
    # Expect: None/NaN each produce a single NaN entry; empty string becomes one empty string
    assert len(out) == (2 + 1 + 1 + 1 + 3 + 1)  # a,b + None + c + NaN + d,e,f + ""
    # Ensure alignment preserved
    assert out.loc[out["keep"] == 5, "col1"].tolist() == ["d", "e", "f"]


def test_split_and_expand_large_empty_df():
    df = pd.DataFrame({"col1": [], "keep": []})
    out = split_and_expand_large(df, split_col="col1", delimiter="|", expand_cols=["keep"])
    assert out.empty


def test_create_nodedf_from_edgedf_basic():
    edges = pd.DataFrame(
        {
            "source_layer": ["A", "A"],
            "source_id": ["s1", "s2"],
            "target_layer": ["B", "A"],
            "target_id": ["t1", "s1"],
        },
    )
    nodes = create_nodedf_from_edgedf(edges)
    # Should include unique pairs from both ends
    assert set(nodes.columns) == {"layer", "node_id"}
    assert set(map(tuple, nodes.values)) == {("A", "s1"), ("A", "s2"), ("B", "t1")}


def test_clean_missing_strings_various_placeholders_and_whitespace():
    df = pd.DataFrame(
        {
            "a": ["  x  ", "NaN", "null", " ", None, 1],
            "b": pd.Series([" y\t", None, "NONE", "z", " w ", "v"], dtype="string"),
        },
    )
    out = clean_missing_strings(df.copy())
    # Stripped values
    assert out.loc[0, "a"] == "x"
    assert out.loc[0, "b"] == "y"
    # Placeholders -> NA
    assert out.loc[1, "a"] is pd.NA or pd.isna(out.loc[1, "a"])  # NaN string -> NA
    assert pd.isna(out.loc[2, "a"])  # null -> NA
    assert pd.isna(out.loc[3, "a"])  # whitespace -> NA
    # Non-string should remain
    assert out.loc[5, "a"] == 1


def test_clean_columns_all_options_and_missing_handling():
    df = pd.DataFrame(
        {
            "x": ["  Hello--world  ", pd.NA, "Foo   Bar"],
            "y": [" CHEBI:123 ", "chebi:456", None],
        },
    )
    out = clean_columns(
        df,
        cols=["x", "y"],
        strip_chars=None,
        trim_substrings=["CHEBI:"],
        lowercase=True,
        collapse_whitespace=True,
        unicode_normalize=True,
        verbose=False,
    )
    assert out.loc[0, "x"] == "hello--world"
    # trim_substrings is case-sensitive; lower-case 'chebi:' is not removed
    assert out.loc[1, "y"] == "chebi:456"

    # Missing column behavior
    out2 = clean_columns(df, cols=["nonexistent"], ignore_missing=True)
    assert list(out2.columns) == ["x", "y"]
    try:
        clean_columns(df, cols=["nonexistent"], ignore_missing=False)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when ignore_missing=False and column absent")


def test_check_for_split_characters_returns_cols(_capsys=None):
    df = pd.DataFrame(
        {
            "a": ["x|y", "z"],
            "b": [1, 2],
            "c": ["no", "split"],
        },
    )
    cols = check_for_split_characters(df, delimiter="|")
    assert "a" in cols
    assert "c" not in cols


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


def test_cache_save_load_and_exists(tmp_path: Path, monkeypatch):
    # Redirect cache root to a temp directory
    monkeypatch.setattr(U, "CACHE_ROOT", tmp_path, raising=True)

    nodes = pd.DataFrame({"node_id": ["n1", "n2"], "layer": ["L", "L"]})
    edges = pd.DataFrame(
        {
            "source_layer": ["L"],
            "source_id": ["n1"],
            "target_layer": ["L"],
            "target_id": ["n2"],
        },
    )

    assert not U.cache_exists("demo")
    U.save_cache("demo", nodes, edges)
    assert U.cache_exists("demo")
    loaded = U.load_cache("demo")
    assert set(loaded.keys()) == {"df_nodes", "df_edges"}
    assert loaded["df_nodes"].equals(nodes)
    assert loaded["df_edges"].equals(edges)
