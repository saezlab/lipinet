#!/usr/bin/env python3
"""
lipinet.parse_reactome.

A standalone module that loads and processes Reactome data into node and edge
DataFrames for LipiNet. Provides a helper function `parse_reactome_data`
and a CLI entrypoint.

Features:
- Robust split of physical entity "name [location]" → ('pe_name', 'pe_location')
- Edge sets:
    * Pathway ontology (child→parent)
    * PathwayID → Pathway ontology
    * ChEBI → Physical Entity
    * Physical Entity → Pathway
    * Physical Entity → Reaction
    * (PE name x loc) → Physical Entity
    * PE name → (PE name x loc)
    * PE loc  → (PE name x loc)
- Node sets for each layer, including ontology nodes
- Optional human-only filtering that preserves rows with unknown species
- Processed caching via lipinet.utils.{cache_exists,load_cache,save_cache}
  (cache key includes species scope: reactome_human / reactome_all)

Returns (dict):
    {
      'df_edges': DataFrame,              # filtered per human_only
      'df_nodes': DataFrame,              # filtered per human_only
      'df_edges_unfiltered': DataFrame,   # convenience extra (not cached)
      'df_nodes_unfiltered': DataFrame,   # convenience extra (not cached)
    }
"""
from __future__ import annotations

import argparse
import importlib

import pandas as pd

import lipinet
import lipinet.databases as db
from lipinet.databases import get_prior_knowledge
from lipinet.utils import (
    cache_exists,
    load_cache,
    save_cache,
)

# Ensure local edits are picked up (mirrors the notebook behavior)
importlib.reload(lipinet)
importlib.reload(db)


# ----------------------------- helpers ------------------------------------ #


def _safe_drop(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Drop columns that exist; ignore the rest."""
    keep = [c for c in df.columns if c not in set(cols)]
    return df[keep]


def _filter_human(df: pd.DataFrame, human_only: bool) -> pd.DataFrame:
    """
    Keep rows where 'human' is not False.

    This preserves rows with NaN in 'human' (e.g., ChEBI, PE name/location).
    If the column is absent, return the DataFrame unchanged.
    """
    if not human_only or "human" not in df.columns:
        return df
    return df[df["human"].ne(False)]


def _split_pe_name_location_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """Match the notebook: strip trailing ']' then split on literal ' [' once."""
    df = df.copy()
    s = df["reactome_pe_name"].astype("string")
    parts = s.str.replace("]", "", regex=False).str.split(" [", n=1, expand=True, regex=False)
    if parts.shape[1] == 1:
        df["pe_name"] = parts[0]
        df["pe_location"] = pd.NA
    else:
        df["pe_name"] = parts[0]
        df["pe_location"] = parts[1]
    return df


def _filter_reactome(df: pd.DataFrame, human_only: bool = True) -> pd.DataFrame:
    """Match the notebook's filter: keep rows where 'human' != False (NaN kept)."""
    if not human_only:
        return df
    if "human" not in df.columns:
        return df
    return df[df["human"].ne(False)]


def parse_reactome_data(
    verbose: bool = False,
    use_cache: bool = False,
    force_download: bool = False,
    human_only: bool = True,
):
    """
    Parse Reactome raw tables into LipiNet nodes and edges.

    Parameters
    ----------
    verbose : bool, optional
        If True, print progress messages (default False).
    use_cache : bool, optional
        If True, load/save the processed cache keyed by species scope (default False).
    force_download : bool, optional
        If True, refetch raw tables even if present locally (default False).
    human_only : bool, optional
        If True, keep only rows where the computed 'human' flag is not False
        (rows with missing 'human' are kept), matching the notebook logic (default True).

    Returns
    -------
    dict
        Dictionary with keys 'df_edges' and 'df_nodes' containing filtered DataFrames.
    """
    cache_key = f"reactome_{'human' if human_only else 'all'}_nb"
    if use_cache and not force_download and cache_exists(cache_key):
        if verbose:
            print(f"↪ loading Reactome (processed) from cache: {cache_key}")
        return load_cache(cache_key)

    # --- Download (same source as notebook) ---
    if verbose:
        print("⏬ loading Reactome raw tables …")
    reactome_dfs = get_prior_knowledge("reactome", verbose=verbose, force_download=force_download)

    # unpack
    df_pe_all = reactome_dfs["ChEBI2Reactome_PE_All_Levels.tsv"].copy()
    df_pe_reac = reactome_dfs["ChEBI2Reactome_PE_Reactions.tsv"].copy()
    df_pathways = reactome_dfs["ReactomePathways.tsv"].copy()
    df_path_rel = reactome_dfs["ReactomePathwaysRelation.tsv"].copy()

    # --- split pe name/location (exact notebook behavior) ---
    df_pe_all = _split_pe_name_location_notebook(df_pe_all)

    # =======================
    # Edges (exactly as NB)
    # =======================

    # pathway hierarchy (ontology child -> parent *within* reactome_pathway)
    df_edges_ontpathway_to_ontpathway = df_path_rel.copy()
    df_edges_ontpathway_to_ontpathway.columns = ["source_id", "target_id"]
    df_edges_ontpathway_to_ontpathway["source_layer"] = "reactome_pathway"
    df_edges_ontpathway_to_ontpathway["target_layer"] = "reactome_pathway"
    df_edges_ontpathway_to_ontpathway["interlayer"] = False

    # pathway nodes (from ReactomePathways table)
    df_nodes_ontpathway = df_pathways.copy()
    df_nodes_ontpathway.columns = ["node_id", "name", "species"]
    df_nodes_ontpathway = (
        df_nodes_ontpathway.assign(layer="reactome_pathway")
        [["layer", "node_id", "name", "species"]]
    )

    # pathwayID -> ontology edges no longer needed:
    # ontology is represented as intra-layer pathway hierarchy above.
    df_edges_pathwayid_to_ontpathway = pd.DataFrame(
        columns=["source_layer", "source_id", "target_layer", "target_id"]
    )

    # ChEBI -> PE
    df_edges_chebi_to_physicalent = (
        df_pe_all[["source_db_identifier", "reactome_pe_stableid"]]
        .drop_duplicates()
        .rename(columns={"source_db_identifier": "source_id", "reactome_pe_stableid": "target_id"})
        .assign(source_layer="reactome_chebi", target_layer="reactome_physicalent")
    )

    # PE -> pathway
    df_edges_phyiscalent_to_pathwayid = df_pe_all.assign(
        source_layer="reactome_physicalent",
        target_layer="reactome_pathway",
    ).rename(columns={
        "reactome_pe_stableid": "source_id",
        "reactome_pathway_stableid": "target_id",
    })
    df_edges_phyiscalent_to_pathwayid["human"] = df_edges_phyiscalent_to_pathwayid["species"].eq("Homo sapiens")

    # PE -> reaction
    df_edges_phyiscalent_to_reactionid = df_pe_reac.assign(
        source_layer="reactome_physicalent",
        target_layer="reactome_reactions",
    ).rename(columns={
        "reactome_pe_stableid": "source_id",
        "reactome_pathway_stableid": "target_id",
    })
    df_edges_phyiscalent_to_reactionid["human"] = df_edges_phyiscalent_to_reactionid["species"].eq("Homo sapiens")

    # (PE name x loc) -> PE
    df_edges_penameloc_to_phyiscalent = df_pe_all.assign(
        source_layer="reactome_physicalent_nameloc",
        target_layer="reactome_physicalent",
    ).rename(columns={
        "reactome_pe_name": "source_id",
        "reactome_pe_stableid": "target_id",
    })[["source_layer", "source_id", "target_layer", "target_id", "pe_name", "pe_location"]].drop_duplicates()

    # PE name -> (PE name x loc)
    df_edges_pename_to_penameloc = df_pe_all.assign(
        source_layer="reactome_physicalent_name",
        target_layer="reactome_physicalent_nameloc",
    ).rename(columns={
        "pe_name": "source_id",
        "reactome_pe_name": "target_id",
    })[["source_layer", "source_id", "target_layer", "target_id"]].drop_duplicates()

    # PE loc -> (PE name x loc)
    df_edges_peloc_to_penameloc = df_pe_all.assign(
        source_layer="reactome_physicalent_loc",
        target_layer="reactome_physicalent_nameloc",
    ).rename(columns={
        "pe_location": "source_id",
        "reactome_pe_name": "target_id",
    })[["source_layer", "source_id", "target_layer", "target_id"]].drop_duplicates()

    # concat edges (same order as notebook; includes ontology edges)
    df_edges_unfiltered = pd.concat(
        [
            df_edges_chebi_to_physicalent,
            df_edges_phyiscalent_to_reactionid,
            df_edges_phyiscalent_to_pathwayid,
            df_edges_pathwayid_to_ontpathway,
            df_edges_penameloc_to_phyiscalent,
            df_edges_pename_to_penameloc,
            df_edges_peloc_to_penameloc,
            df_edges_ontpathway_to_ontpathway,
        ],
        ignore_index=True,
    ).drop_duplicates()

    # =======================
    # Nodes (exactly as NB)
    # =======================

    # chebi
    df_nodes_chebi = df_edges_chebi_to_physicalent.drop(
        columns=["target_layer", "target_id"]
    ).rename(columns={"source_id": "node_id", "source_layer": "layer"}).drop_duplicates()

    # pathwayid (same objects as df_nodes_ontpathway; add human flag)
    df_nodes_pathwayid = df_nodes_ontpathway.copy()
    df_nodes_pathwayid["human"] = df_nodes_pathwayid["species"].eq("Homo sapiens")

    # physicalent
    df_nodes_physicalent = df_edges_phyiscalent_to_pathwayid.drop(
        columns=["target_layer", "target_id"]
    ).rename(columns={"source_id": "node_id", "source_layer": "layer"}) \
     .drop(columns=["event_name_pathway_or_reaction", "url", "evidence_code"], errors="ignore") \
     .drop_duplicates()
    df_nodes_physicalent["human"] = df_nodes_physicalent["species"].eq("Homo sapiens")

    # reactionid
    df_nodes_reactionid = df_edges_phyiscalent_to_reactionid.drop(
        columns=["source_layer", "source_id"]
    ).rename(columns={"target_id": "node_id", "target_layer": "layer"}) \
     .drop(columns=["source_db_identifier", "reactome_pe_name"], errors="ignore") \
     .drop_duplicates()

    # penameloc
    df_nodes_penameloc = df_edges_penameloc_to_phyiscalent.drop(
        columns=["target_layer", "target_id"]
    ).rename(columns={"source_id": "node_id", "source_layer": "layer"}).drop_duplicates()

    # pename
    df_nodes_pename = df_edges_pename_to_penameloc.drop(
        columns=["target_layer", "target_id"]
    ).rename(columns={"source_id": "node_id", "source_layer": "layer"}).drop_duplicates()

    # peloc
    df_nodes_peloc = df_edges_peloc_to_penameloc.drop(
        columns=["target_layer", "target_id"]
    ).rename(columns={"source_id": "node_id", "source_layer": "layer"}).drop_duplicates()

    # concat nodes (same order as notebook)
    df_nodes_unfiltered = pd.concat(
        [
            df_nodes_chebi,
            df_nodes_physicalent,
            df_nodes_reactionid,
            df_nodes_pathwayid,  # already includes pathway nodes
            df_nodes_penameloc,
            df_nodes_pename,
            df_nodes_peloc,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates()

    # --- human-only filter (same logic as notebook) ---
    df_nodes = _filter_reactome(df_nodes_unfiltered, human_only=human_only)
    df_edges = _filter_reactome(df_edges_unfiltered, human_only=human_only)

    if verbose:
        print(f"[reactome] edges: {df_edges.shape}")
        print(f"[reactome] nodes: {df_nodes.shape}")

    result = {"df_edges": df_edges, "df_nodes": df_nodes}

    if use_cache:
        if verbose:
            print(f"↪ caching Reactome (processed) as {cache_key}")
        save_cache(cache_key, df_nodes=result["df_nodes"], df_edges=result["df_edges"])

    return result


def main():
    """CLI entry point for parsing Reactome data."""
    p = argparse.ArgumentParser(description="Parse Reactome exactly like the exploration notebook.")
    p.add_argument("--quiet", action="store_true", help="suppress prints")
    p.add_argument("--use-cache", action="store_true", help="load/save processed cache")
    p.add_argument("--force-download", action="store_true", help="force fresh raw download")
    p.add_argument("--all-species", action="store_true", help="keep all species (no human-only filter)")
    args = p.parse_args()

    out = parse_reactome_data(
        verbose=not args.quiet,
        use_cache=args.use_cache,
        force_download=args.force_download,
        human_only=not args.all_species,
    )
    if not args.quiet:
        print("done.", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
