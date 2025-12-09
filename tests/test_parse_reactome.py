import sys

import pandas as pd

import lipinet.parse_reactome as pr


def _reactome_minimal_dfs():
    # Minimal Reactome tables after databases.clean() renaming
    df_pe_all = pd.DataFrame(
        {
            "source_db_identifier": ["CHEBI:1"],
            "reactome_pe_stableid": ["PE1"],
            "reactome_pe_name": ["Glucose [cytosol]"],
            "reactome_pathway_stableid": ["P1"],
            "url": ["u"],
            "event_name_pathway_or_reaction": ["name"],
            "evidence_code": ["ECO"],
            "species": ["Homo sapiens"],
        },
    )

    df_pe_reac = pd.DataFrame(
        {
            "reactome_pe_stableid": ["PE1", "PE2"],
            # In this table, target gets treated as reaction IDs in the parser
            "reactome_pathway_stableid": ["R1", "R2"],
            "species": ["Homo sapiens", "Mus musculus"],
        },
    )

    df_pathways = pd.DataFrame(
        {
            "reactome_pathway_stableid": ["P0", "P1"],
            "reactome_pathway_name": ["root", "child"],
            "species": ["Homo sapiens", "Homo sapiens"],
        },
    )

    df_path_rel = pd.DataFrame(
        {
            "parent_stableid": ["P0"],
            "child_stableid": ["P1"],
        },
    )

    return {
        "ChEBI2Reactome_PE_All_Levels.tsv": df_pe_all,
        "ChEBI2Reactome_PE_Reactions.tsv": df_pe_reac,
        "ReactomePathways.tsv": df_pathways,
        "ReactomePathwaysRelation.tsv": df_path_rel,
    }


def test_parse_reactome_data_basic(monkeypatch):
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: _reactome_minimal_dfs())

    out = pr.parse_reactome_data(verbose=True, use_cache=False, force_download=False)
    edges, nodes = out["df_edges"], out["df_nodes"]

    # Basic shape
    assert {"source_layer", "source_id", "target_layer", "target_id"}.issubset(edges.columns)
    assert {"layer", "node_id"}.issubset(nodes.columns)

    # ChEBI -> Physical Entity edge exists
    mask = (
        (edges["source_layer"] == "reactome_chebi")
        & (edges["target_layer"] == "reactome_physicalent")
        & (edges["source_id"] == "CHEBI:1")
        & (edges["target_id"] == "PE1")
    )
    assert mask.any()

    # Physical Entity -> Pathway edge exists
    assert (
        (edges["source_layer"] == "reactome_physicalent")
        & (edges["target_layer"] == "reactome_pathway")
    ).any()

    # Pathway ontology edges exist (pathway→pathway)
    assert (
        (edges["source_layer"] == "reactome_pathway")
        & (edges["target_layer"] == "reactome_pathway")
    ).any()

    # Nodes include expected layers
    assert {
        "reactome_pathway",
        "reactome_physicalent",
        "reactome_reactions",
    }.issubset(set(nodes["layer"]))


def test_parse_reactome_human_filter(monkeypatch):
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: _reactome_minimal_dfs())
    out = pr.parse_reactome_data(
        verbose=False, use_cache=False, force_download=False, human_only=True
    )
    edges = out["df_edges"]

    # Reaction edges should only include the human one (R1), mouse (R2) filtered out
    react_edges = edges[edges["target_layer"] == "reactome_reactions"]
    assert set(react_edges["target_id"]) == {"R1"}


def test_parse_reactome_uses_cache(monkeypatch):
    # Force cache branch
    monkeypatch.setattr(pr, "cache_exists", lambda _key: True)
    cached = {"df_nodes": pd.DataFrame(), "df_edges": pd.DataFrame()}
    monkeypatch.setattr(pr, "load_cache", lambda _key: cached)
    out = pr.parse_reactome_data(verbose=True, use_cache=True, force_download=False)
    assert out is cached


def test_parse_reactome_saves_cache(monkeypatch):
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: _reactome_minimal_dfs())
    called = {"n": 0}
    monkeypatch.setattr(
        pr, "save_cache", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1)
    )
    pr.parse_reactome_data(verbose=False, use_cache=True, force_download=True)
    assert called["n"] == 1


def test_parse_reactome_main_quiet(monkeypatch):
    monkeypatch.setattr(pr, "get_prior_knowledge", lambda *_a, **_k: _reactome_minimal_dfs())
    argv = sys.argv
    try:
        sys.argv = ["prog", "--quiet"]
        pr.main()
    finally:
        sys.argv = argv
