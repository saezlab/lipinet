#!/usr/bin/env python3
"""
lipinet.build_lipinet

Builds the combined LipiNet graph (nodes & edges) from parsed sources
(SwissLipids, Rhea), including cross-source linking (e.g., ChEBI).

Provides:
  - build_lipinet_data(verbose=False, use_cache=False, force_download=False)
  - CLI entrypoint: python -m lipinet.build_lipinet [--use-cache] [--force-download] [--quiet] [--save]

Cache semantics:
  - If use_cache is True and cache for 'lipinet' exists (and not force_download),
    load and return.
  - Otherwise build fresh; if use_cache True, write cache at the end.

Output (when --save given):
  - .data/processed/lipinet_nodes.parquet
  - .data/processed/lipinet_edges.parquet
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from lipinet.parse_swisslipids import parse_swisslipids_data
from lipinet.parse_rhea import parse_rhea_data
from lipinet.utils import clean_missing_strings

# Reuse the generic cache helpers you added earlier
from lipinet.utils import save_cache, load_cache, cache_exists

# Where to write optional outputs
DATA_PROCESSED = Path(__file__).parent / ".data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Helpers
# ---------------------------

def _trim_chebi(series: pd.Series) -> pd.Series:
    """Return numeric/ID part by stripping a leading 'CHEBI:' if present."""
    s = series.astype(str).str.strip()
    return s.str.replace("CHEBI:", "", regex=False)

def _link_chebi_edges(df_sl_nodes: pd.DataFrame, df_rhea_nodes: pd.DataFrame, verbose: bool=False) -> pd.DataFrame:
    """
    Create interlayer edges between SwissLipids sl_chebi and Rhea rhea_chebiid
    by matching identical ChEBI IDs (SwissLipids uses plain IDs, Rhea uses 'CHEBI:ID').
    """
    sl_chebi = df_sl_nodes[df_sl_nodes["layer"] == "sl_chebi"].copy()
    rhea_chebi = df_rhea_nodes[df_rhea_nodes["layer"] == "rhea_chebiid"].copy()

    # Minimal, safe normalization on just node_id
    sl_chebi = sl_chebi.dropna(subset=["node_id"])
    rhea_chebi = rhea_chebi.dropna(subset=["node_id"])
    sl_chebi["node_id"] = sl_chebi["node_id"].astype(str).str.strip()
    rhea_chebi["node_id"] = rhea_chebi["node_id"].astype(str).str.strip()

    # Prepare Rhea trimmed IDs for matching
    rhea_chebi["node_id_trimmed"] = _trim_chebi(rhea_chebi["node_id"])

    # Build a tidy edge table
    df_sl = sl_chebi[["layer", "node_id"]].rename(columns={"layer":"source_layer", "node_id":"source_id"})
    df_rh = rhea_chebi[["layer", "node_id", "node_id_trimmed"]].rename(columns={"layer":"target_layer", "node_id":"target_id"})

    merged = df_sl.merge(df_rh, left_on="source_id", right_on="node_id_trimmed", how="inner")
    edges = merged[["source_layer", "source_id", "target_layer", "target_id"]].drop_duplicates()

    edges = edges.assign(
        interlayer=True,
        edge_type="same_id_chebi",
        origin_edge="lipinet",
    )

    if verbose:
        print(f"Linked ChEBI edges: {edges.shape[0]}")

    return edges


def _join_node_dfs(df_sl_nodes: pd.DataFrame, df_rhea_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Combine node frames from SwissLipids and Rhea, tagging origin and
    prefixing source-unique columns.
    """
    df_sl = df_sl_nodes.copy()
    df_rh = df_rhea_nodes.copy()

    df_sl["origin_vertex"] = "swisslipids"
    df_rh["origin_vertex"] = "rhea"

    # shared vs unique
    common = df_rh.columns.intersection(df_sl.columns)
    unique_rhea = df_rh.columns.difference(common)
    unique_sl = df_sl.columns.difference(common)

    df_rhea_pref = df_rh.rename(columns={c: f"rhea_{c}" for c in unique_rhea})
    df_sl_pref = df_sl.rename(columns={c: f"sl_{c}" for c in unique_sl})

    df_nodes = pd.concat([df_rhea_pref, df_sl_pref], ignore_index=True, sort=False)

    # put shared + origin first
    all_cols = df_nodes.columns.tolist()
    shared_cols = [c for c in all_cols if not c.startswith(("rhea_", "sl_"))]
    prefixed_cols = [c for c in all_cols if c.startswith(("rhea_", "sl_"))]
    df_nodes = df_nodes[shared_cols + prefixed_cols]

    # Basic cleanup
    df_nodes = clean_missing_strings(df_nodes)
    df_nodes = df_nodes.drop_duplicates(subset=["layer", "node_id"])

    return df_nodes


def _join_edge_dfs(df_sl_edges: pd.DataFrame, df_rhea_edges: pd.DataFrame, df_chebi_linked: pd.DataFrame) -> pd.DataFrame:
    """
    Stack edges from sources and the cross-source links; add origin labels.
    """
    sl = df_sl_edges.copy().assign(origin_edge="swisslipids")
    rh = df_rhea_edges.copy().assign(origin_edge="rhea")

    df_edges = pd.concat([rh, sl, df_chebi_linked], ignore_index=True, sort=False)
    df_edges = clean_missing_strings(df_edges).drop_duplicates()

    # Reorder a few key columns to the front if they exist
    front = [c for c in ["source_layer","source_id","target_layer","target_id","interlayer","edge_type","origin_edge"] if c in df_edges.columns]
    df_edges = df_edges[front + [c for c in df_edges.columns if c not in front]]

    return df_edges


# ---------------------------
# Public API
# ---------------------------

def build_lipinet_data(verbose: bool=False, use_cache: bool=False, force_download: bool=False) -> dict[str, pd.DataFrame]:
    """
    Build combined LipiNet nodes & edges from SwissLipids and Rhea.

    Cache behavior mirrors parse_* modules:
      - If use_cache and not force_download and cache exists('lipinet'): load & return.
      - Else build; if use_cache, save to cache before returning.
    """
    # Load from cache?
    if use_cache and not force_download and cache_exists("lipinet"):
        if verbose:
            print("↪ Loading LipiNet (combined) from cache")
        return load_cache("lipinet")

    # Parse source datasets (these can use their own cache settings)
    sl = parse_swisslipids_data(verbose=verbose, use_cache=use_cache, force_download=force_download)
    rhea = parse_rhea_data(verbose=verbose, use_cache=use_cache, force_download=force_download) 

    df_sl_nodes = sl["df_nodes"].copy()
    df_sl_edges = sl["df_edges"].copy()
    df_rhea_nodes = rhea["df_nodes"].copy()
    df_rhea_edges = rhea["df_edges"].copy()

    # Remove literal "nan" strings if any slipped through
    df_sl_nodes = df_sl_nodes[df_sl_nodes["node_id"].astype(str) != "nan"].copy()

    # Cross-source link edges (ChEBI)
    df_chebi_linked = _link_chebi_edges(df_sl_nodes, df_rhea_nodes, verbose=verbose)

    # Combine nodes & edges
    df_nodes = _join_node_dfs(df_sl_nodes, df_rhea_nodes)
    df_edges = _join_edge_dfs(df_sl_edges, df_rhea_edges, df_chebi_linked)

    result = {"df_nodes": df_nodes, "df_edges": df_edges}

    # Save cache?
    if use_cache:
        if verbose:
            print("↪ Caching LipiNet (combined) nodes & edges")
        save_cache("lipinet", df_nodes=df_nodes, df_edges=df_edges)

    if verbose:
        print(f"Built LipiNet: {df_nodes.shape[0]} nodes, {df_edges.shape[0]} edges")

    return result


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Build combined LipiNet (SwissLipids + Rhea)")
    parser.add_argument("--use-cache", action="store_true", help="Load/save combined result from cache")
    parser.add_argument("--force-download", action="store_true", help="Force fresh fetch/parse of sources")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--save", action="store_true", help="Also write Parquet files to .data/processed/")
    args = parser.parse_args()

    verbose = not args.quiet
    res = build_lipinet_data(verbose=verbose, use_cache=args.use_cache, force_download=args.force_download)

    if args.save:
        nodes_path = DATA_PROCESSED / "lipinet_nodes.parquet"
        edges_path = DATA_PROCESSED / "lipinet_edges.parquet"
        res["df_nodes"].to_parquet(nodes_path, index=False)
        res["df_edges"].to_parquet(edges_path, index=False)
        if verbose:
            print(f"Wrote: {nodes_path}")
            print(f"Wrote: {edges_path}")

if __name__ == "__main__":
    main()