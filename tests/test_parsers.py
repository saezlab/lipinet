import sys

import pandas as pd

import lipinet.parse_rhea as pr
import lipinet.parse_swisslipids as ps


def test_parse_swisslipids_data_basic(monkeypatch):
    # Stub SwissLipids resource with minimal required columns
    df = pd.DataFrame(
        {
            "Lipid ID": ["L1", "L2"],
            "CHEBI": ["1|2", None],
            "LIPID MAPS": [None, "LM:1"],
            "HMDB": [None, None],
            "MetaNetX": [None, None],
            "PMID": [None, None],
            "Lipid class*": ["ClassA", "ClassB"],
            "Abbreviation*": [None, None],
            "Synonyms*": [None, None],
            "Parent": [None, None],
            "Components*": ["FA(16:0)/FA(18:1)", None],
        },
    )

    monkeypatch.setattr(ps, "get_prior_knowledge", lambda _name, **_kw: df)

    res = ps.parse_swisslipids_data(verbose=True, force_download=False, use_cache=False)
    nodes, edges = res["df_nodes"], res["df_edges"]
    # basic shape & invariants
    assert {"source_layer", "source_id", "target_layer", "target_id", "interlayer"}.issubset(
        edges.columns,
    )
    assert {"layer", "node_id"}.issubset(nodes.columns)
    # Ensure multilinks split for CHEBI and parsed components
    assert (edges["target_layer"] == "sl_chebi").any()
    assert (edges["target_layer"].str.contains("sl_components")).any()


def test_process_ec_numbers_and_edges_nodes():
    base = pd.DataFrame(
        {
            "EC number": ["EC:1.2.3.4;EC:2.3.4.5"],
        },
    )
    df_ec = pr.process_ec_numbers(base)
    assert {"Main_Class", "Subclass", "Subsubclass", "Serial_Number", "EC_number"}.issubset(
        df_ec.columns,
    )
    edges, nodes = pr.build_rhea_ec_edges_and_nodes(df_ec)
    assert {"source_id", "target_id", "source_layer", "target_layer", "interlayer"}.issubset(
        edges.columns,
    )
    assert {"node_id", "layer", "ec_level"}.issubset(nodes.columns)


def test_explode_columns_and_parse_rhea_data(monkeypatch):
    df = pd.DataFrame(
        {
            "Reaction identifier": ["R1"],
            "Equation": ["A=B"],
            "ChEBI identifier": ["CHEBI:1;CHEBI:2"],
            "ChEBI name": ["A;B"],
            "EC number": ["EC:1.2.3.4;EC:2.3.4.5"],
            "Enzymes": ["E1"],
            "Gene Ontology": ["GO:1"],
            "Cross-reference (Reactome)": ["RXN:1"],
        },
    )
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda _name, **_kw: df)
    res = pr.parse_rhea_data(verbose=True, use_cache=False, force_download=False)
    edges, nodes = res["df_edges"], res["df_nodes"]
    # Edge types present
    assert set(edges["edge_type"]).issuperset({"ec_hierarchy", "reaction_chebi", "reaction_ec"})
    # Node layers present
    assert {"rhea_reactionid", "rhea_chebiid", "rhea_ec"}.issubset(set(nodes["layer"]))


def test_parse_swisslipids_uses_cache(monkeypatch):
    # Force cache branch
    monkeypatch.setattr(ps, "cache_exists", lambda _name: True)
    cached = {"df_nodes": pd.DataFrame(), "df_edges": pd.DataFrame()}
    monkeypatch.setattr(ps, "load_cache", lambda _name: cached)
    out = ps.parse_swisslipids_data(verbose=True, use_cache=True, force_download=False)
    assert out is cached


def test_parse_rhea_fallback_and_cache(monkeypatch):
    # First exercise the early cache branch
    monkeypatch.setattr(pr, "cache_exists", lambda _name: True)
    cached = {"df_nodes": pd.DataFrame(), "df_edges": pd.DataFrame()}
    monkeypatch.setattr(pr, "load_cache", lambda _name: cached)
    out = pr.parse_rhea_data(verbose=True, use_cache=True, force_download=False)
    assert out is cached

    # Now exercise the fallback path by raising in get_prior_knowledge
    monkeypatch.setattr(pr, "cache_exists", lambda _name: False)
    monkeypatch.setattr(
        pr,
        "get_prior_knowledge",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    # Monkeypatch pandas.read_csv for the fallback path
    monkeypatch.setattr(
        pr.pd,
        "read_csv",
        lambda *_a, **_k: pd.DataFrame(
            {
                "Reaction identifier": ["R1"],
                "Equation": ["A=B"],
                "ChEBI identifier": ["CHEBI:1;CHEBI:2"],
                "ChEBI name": ["A;B"],
                "EC number": ["EC:1.2.3.4;EC:2.3.4.5"],
                "Enzymes": ["E1"],
                "Gene Ontology": ["GO:1"],
                "Cross-reference (Reactome)": ["RXN:1"],
            },
        ),
    )

    res = pr.parse_rhea_data(verbose=True, use_cache=False, force_download=False)
    assert set(res.keys()) == {"df_edges", "df_nodes"}


def test_parse_rhea_main_quiet(monkeypatch):
    df = pd.DataFrame(
        {
            "Reaction identifier": ["R1"],
            "Equation": ["A=B"],
            "ChEBI identifier": ["CHEBI:1;CHEBI:2"],
            "ChEBI name": ["A;B"],
            "EC number": ["EC:1.2.3.4;EC:2.3.4.5"],
            "Enzymes": ["E1"],
            "Gene Ontology": ["GO:1"],
            "Cross-reference (Reactome)": ["RXN:1"],
        },
    )
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: df)
    argv = sys.argv
    try:
        sys.argv = ["prog", "--quiet"]
        pr.main()
    finally:
        sys.argv = argv


def test_parse_swisslipids_saves_cache(monkeypatch):
    # Minimal DF
    df = pd.DataFrame(
        {
            "Lipid ID": ["L1"],
            "CHEBI": ["1"],
            "LIPID MAPS": [None],
            "HMDB": [None],
            "MetaNetX": [None],
            "PMID": [None],
            "Lipid class*": ["ClassA"],
            "Abbreviation*": [None],
            "Synonyms*": [None],
            "Parent": [None],
            "Components*": ["FA(16:0)/FA(18:1)"],
        },
    )
    monkeypatch.setattr(ps, "get_prior_knowledge", lambda *_a, **_k: df)
    called = {"n": 0}
    monkeypatch.setattr(
        ps, "save_cache", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1)
    )
    ps.parse_swisslipids_data(verbose=True, use_cache=True, force_download=True)
    assert called["n"] == 1


def test_parse_rhea_saves_cache(monkeypatch):
    df = pd.DataFrame(
        {
            "Reaction identifier": ["R1"],
            "Equation": ["A=B"],
            "ChEBI identifier": ["CHEBI:1;CHEBI:2"],
            "ChEBI name": ["A;B"],
            "EC number": ["EC:1.2.3.4;EC:2.3.4.5"],
            "Enzymes": ["E1"],
            "Gene Ontology": ["GO:1"],
            "Cross-reference (Reactome)": ["RXN:1"],
        },
    )
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: df)
    called = {"n": 0}
    monkeypatch.setattr(
        pr, "save_cache", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1)
    )
    pr.parse_rhea_data(verbose=True, use_cache=True, force_download=True)
    assert called["n"] == 1
