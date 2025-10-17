import sys

import pandas as pd

import lipinet.build_lipinet as bl


def test_build_lipinet_data_combines_and_links(monkeypatch):
    # SwissLipids stub
    sl_nodes = pd.DataFrame({
        "node_id": ["1", "X"],
        "layer": ["sl_chebi", "sl_other"],
    })
    sl_edges = pd.DataFrame({
        "source_layer": ["sl_chebi"],
        "source_id": ["1"],
        "target_layer": ["sl_other"],
        "target_id": ["X"],
        "interlayer": [True],
    })

    # Rhea stub (note CHEBI: prefix)
    rh_nodes = pd.DataFrame({
        "node_id": ["CHEBI:1", "Y"],
        "layer": ["rhea_chebiid", "rhea_other"],
    })
    rh_edges = pd.DataFrame({
        "source_layer": ["rhea_other"],
        "source_id": ["Y"],
        "target_layer": ["rhea_chebiid"],
        "target_id": ["CHEBI:1"],
        "interlayer": [True],
    })

    monkeypatch.setattr(bl, "parse_swisslipids_data", lambda **kw: {"df_nodes": sl_nodes, "df_edges": sl_edges})
    monkeypatch.setattr(bl, "parse_rhea_data", lambda **kw: {"df_nodes": rh_nodes, "df_edges": rh_edges})

    res = bl.build_lipinet_data(verbose=True, use_cache=False, force_download=False)
    nodes, edges = res["df_nodes"], res["df_edges"]
    # Cross-link edge should exist between sl_chebi 1 and rhea_chebiid CHEBI:1
    mask = (
        (edges["edge_type"] == "same_id_chebi") &
        (edges["source_layer"] == "sl_chebi") &
        (edges["target_layer"] == "rhea_chebiid")
    )
    assert mask.any()
    # Nodes combined and deduped
    assert {"origin_vertex"}.issubset(nodes.columns)


def test_build_lipinet_uses_cache(monkeypatch):
    # Ensure builder early cache path
    monkeypatch.setattr(bl, "cache_exists", lambda name: True)
    cached = {"df_nodes": pd.DataFrame(), "df_edges": pd.DataFrame()}
    monkeypatch.setattr(bl, "load_cache", lambda name: cached)
    out = bl.build_lipinet_data(verbose=True, use_cache=True, force_download=False)
    assert out is cached


def test_build_lipinet_saves_cache(monkeypatch):
    sl_nodes = pd.DataFrame({"node_id": ["1"], "layer": ["sl_chebi"]})
    sl_edges = pd.DataFrame({
        "source_layer": ["sl_chebi"],
        "source_id": ["1"],
        "target_layer": ["sl_chebi"],
        "target_id": ["1"],
        "interlayer": [False],
    })
    rh_nodes = pd.DataFrame({"node_id": ["CHEBI:1"], "layer": ["rhea_chebiid"]})
    rh_edges = pd.DataFrame({
        "source_layer": ["rhea_chebiid"],
        "source_id": ["CHEBI:1"],
        "target_layer": ["rhea_chebiid"],
        "target_id": ["CHEBI:1"],
        "interlayer": [False],
    })
    monkeypatch.setattr(bl, "parse_swisslipids_data", lambda **kw: {"df_nodes": sl_nodes, "df_edges": sl_edges})
    monkeypatch.setattr(bl, "parse_rhea_data", lambda **kw: {"df_nodes": rh_nodes, "df_edges": rh_edges})
    called = {"n": 0}
    monkeypatch.setattr(bl, "save_cache", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    res = bl.build_lipinet_data(verbose=True, use_cache=True, force_download=True)
    assert set(res.keys()) == {"df_nodes", "df_edges"}
    assert called["n"] == 1


def test_build_lipinet_main_save(monkeypatch):
    sl_nodes = pd.DataFrame({"node_id": ["1"], "layer": ["sl_chebi"]})
    sl_edges = pd.DataFrame({
        "source_layer": ["sl_chebi"],
        "source_id": ["1"],
        "target_layer": ["sl_chebi"],
        "target_id": ["1"],
        "interlayer": [False],
    })
    rh_nodes = pd.DataFrame({"node_id": ["CHEBI:1"], "layer": ["rhea_chebiid"]})
    rh_edges = pd.DataFrame({
        "source_layer": ["rhea_chebiid"],
        "source_id": ["CHEBI:1"],
        "target_layer": ["rhea_chebiid"],
        "target_id": ["CHEBI:1"],
        "interlayer": [False],
    })
    monkeypatch.setattr(bl, "parse_swisslipids_data", lambda **kw: {"df_nodes": sl_nodes, "df_edges": sl_edges})
    monkeypatch.setattr(bl, "parse_rhea_data", lambda **kw: {"df_nodes": rh_nodes, "df_edges": rh_edges})
    # Avoid requiring pyarrow/fastparquet
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, index=False: None, raising=False)

    argv = sys.argv
    try:
        sys.argv = ["prog", "--save", "--quiet"]
        bl.main()
    finally:
        sys.argv = argv


def test_build_lipinet_main_saves_and_prints(monkeypatch, capsys):
    # Same stubs as before
    sl_nodes = pd.DataFrame({"node_id": ["1"], "layer": ["sl_chebi"]})
    sl_edges = pd.DataFrame({
        "source_layer": ["sl_chebi"],
        "source_id": ["1"],
        "target_layer": ["sl_chebi"],
        "target_id": ["1"],
        "interlayer": [False],
    })
    rh_nodes = pd.DataFrame({"node_id": ["CHEBI:1"], "layer": ["rhea_chebiid"]})
    rh_edges = pd.DataFrame({
        "source_layer": ["rhea_chebiid"],
        "source_id": ["CHEBI:1"],
        "target_layer": ["rhea_chebiid"],
        "target_id": ["CHEBI:1"],
        "interlayer": [False],
    })
    monkeypatch.setattr(bl, "parse_swisslipids_data", lambda **kw: {"df_nodes": sl_nodes, "df_edges": sl_edges})
    monkeypatch.setattr(bl, "parse_rhea_data", lambda **kw: {"df_nodes": rh_nodes, "df_edges": rh_edges})
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, index=False: None, raising=False)

    argv = sys.argv
    try:
        sys.argv = ["prog", "--save"]  # not quiet: exercise print lines
        bl.main()
        out = capsys.readouterr().out
        assert "Wrote:" in out
    finally:
        sys.argv = argv
