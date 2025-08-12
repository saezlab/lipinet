#!/usr/bin/env python3
"""
lipinet.parse_rhea

A standalone module that loads and processes Rhea data into node and edge DataFrames
for LipiNet. Provides a helper function `parse_rhea_data` and a CLI entrypoint.
"""
import argparse
import pandas as pd
from lipinet.databases import get_prior_knowledge
from lipinet.utils import (
    split_and_expand_large,
    create_nodedf_from_edgedf,
    check_for_split_characters,
    save_cache,
    load_cache,
    cache_exists,
    clean_missing_strings,
)

def process_ec_numbers(df):
    """
    Process the 'EC number' column of the input DataFrame.

    Parameters:
        df (pd.DataFrame): A DataFrame containing an 'EC number' column.

    Returns:
        pd.DataFrame: A new DataFrame with the following columns:
            - 'EC_number': The reassembled EC number in the format 'EC:Main_Class.Subclass.Subsubclass.Serial_Number'
            - 'Main_Class': The first part of the EC number.
            - 'Subclass': The second part of the EC number.
            - 'Subsubclass': The third part of the EC number.
            - 'Serial_Number': The fourth part of the EC number.
    """
    # Split, explode, deduplicate, clean and split by '.' into a Series of lists.
    ec_num_series = df['EC number'].str.split(';')\
                        .explode()\
                        .drop_duplicates()\
                        .dropna()\
                        .str.strip('EC:')\
                        .str.split('.')
    
    # Convert the Series of lists into a DataFrame with named columns.
    df_ec = pd.DataFrame(ec_num_series.tolist(), 
                         columns=["Main_Class", "Subclass", "Subsubclass", "Serial_Number"])
    
    # Create a new column with the reassembled EC number in the original format.
    df_ec.insert(0, 'EC_number', 
                 'EC:' + df_ec["Main_Class"].astype(str) + '.' +
                 df_ec["Subclass"].astype(str) + '.' +
                 df_ec["Subsubclass"].astype(str) + '.' +
                 df_ec["Serial_Number"].astype(str))
    
    return df_ec

def build_rhea_ec_edges_and_nodes(df_ec: pd.DataFrame):
    """
    Given a DataFrame with EC hierarchy columns:
        Main_Class, Subclass, Subsubclass, EC_number,
    this function creates:
      - A DataFrame of edges linking each hierarchical level.
      - A DataFrame of unique nodes with a 'ec_level' column 
        indicating the node's level in the hierarchy.
    """

    # -- Make a copy so we don't modify the original df in-place
    df = df_ec.copy()

    # 1) Build the node representations for each hierarchy level
    #    (convert to string just in case they're numeric)
    df["main_node"] = "EC:" + df["Main_Class"].astype(str)
    df["subclass_node"] = (
        "EC:" + df["Main_Class"].astype(str) 
        + "." + df["Subclass"].astype(str)
    )
    df["subsubclass_node"] = (
        "EC:" + df["Main_Class"].astype(str) 
        + "." + df["Subclass"].astype(str)
        + "." + df["Subsubclass"].astype(str)
    )

    # 2) Create edges for each hierarchical link and label them
    edges1 = df[["main_node", "subclass_node"]].rename(
        columns={"main_node": "source_id", "subclass_node": "target_id"}
    )
    edges1["ec_level"] = "main_class->subclass"

    edges2 = df[["subclass_node", "subsubclass_node"]].rename(
        columns={"subclass_node": "source_id", "subsubclass_node": "target_id"}
    )
    edges2["ec_level"] = "subclass->subsubclass"

    edges3 = df[["subsubclass_node", "EC_number"]].rename(
        columns={"subsubclass_node": "source_id", "EC_number": "target_id"}
    )
    edges3["ec_level"] = "subsubclass->EC_number"

    # Concatenate edges and drop duplicates
    edges_df = pd.concat([edges1, edges2, edges3], ignore_index=True).drop_duplicates()

    # Add the additional columns for your graph structure
    edges_df["source_layer"] = "rhea_ec"
    edges_df["target_layer"] = "rhea_ec"
    edges_df["interlayer"] = False

    # 3) Build a node DataFrame from all unique source/target IDs
    nodes_df = pd.DataFrame(
        pd.concat([edges_df["source_id"], edges_df["target_id"]]).unique(),
        columns=["node_id"]
    )
    nodes_df["layer"] = "rhea_ec"  # same layer for all

    # 4) Build a small lookup table to label each node with ec_level
    #    We'll mark each node as either "main_class", "subclass", "subsubclass", or "full_ec".
    df_main = (
        df[["main_node"]].drop_duplicates().rename(columns={"main_node": "node_id"})
    )
    df_main["ec_level"] = "main_class"

    df_sub = (
        df[["subclass_node"]].drop_duplicates().rename(columns={"subclass_node": "node_id"})
    )
    df_sub["ec_level"] = "subclass"

    df_subsub = (
        df[["subsubclass_node"]].drop_duplicates().rename(columns={"subsubclass_node": "node_id"})
    )
    df_subsub["ec_level"] = "subsubclass"

    df_full = (
        df[["EC_number"]].drop_duplicates().rename(columns={"EC_number": "node_id"})
    )
    df_full["ec_level"] = "full_ec"

    df_nodes_level = pd.concat([df_main, df_sub, df_subsub, df_full], ignore_index=True)
    df_nodes_level.drop_duplicates(subset="node_id", inplace=True)

    # 5) Merge the node-level info into the nodes_df
    nodes_df = nodes_df.merge(df_nodes_level, on="node_id", how="left")

    # Return the final DataFrames
    return edges_df, nodes_df

def explode_columns(df, columns, delimiter=";"):
    """
    Split and explode the specified columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): List of column names to split by the delimiter.
        delimiter (str): The delimiter to use when splitting the column values.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns exploded.

    Note:
        Each row in the specified columns must produce lists of the same length.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        df[col] = df[col].str.split(delimiter)
    return df.explode(columns)

def parse_rhea_data(verbose: bool=False, use_cache: bool=False, force_download: bool=False):
    """
    Core function to load and process Rhea data.

    Parameters:
        verbose (bool): If True, prints detailed status.
        use_cache (bool): If True, load/save processed nodes & edges.
        force_download (bool): If True, refetch raw Rhea and rebuild (ignore cache).

    Returns:
        dict: {'df_edges': DataFrame, 'df_nodes': DataFrame}
    """
    # ---- processed cache (nodes/edges) ----
    if use_cache and not force_download and cache_exists("rhea"):
        if verbose:
            print("↪ Loading Rhea (processed) from cache")
        return load_cache("rhea")

    # Load Rhea data (use get_prior_knowledge if available)
    try:
        df_rhea = get_prior_knowledge('rhea', verbose=verbose, force_download=force_download)
    except Exception:
        # fallback to local TSV if your helper isn't available
        df_rhea = pd.read_csv('../.data/Rhea-.tsv', sep='\t')
    if verbose:
        print(f"Loaded Rhea data: {df_rhea.shape[0]} rows, {df_rhea.shape[1]} columns")

    # Enzyme classification
    df_ec = process_ec_numbers(df_rhea)
    df_edges_ec, df_nodes_ec = build_rhea_ec_edges_and_nodes(df_ec)
    if verbose:
        print(f"EC hierarchy: {df_edges_ec.shape[0]} edges, {df_nodes_ec.shape[0]} nodes")

    # Reaction ↔ ChEBI edges
    df_rhea_exploded = explode_columns(df_rhea, ["ChEBI identifier", "ChEBI name"])
    df_edges_reaction_chebi = (
        df_rhea_exploded[['Reaction identifier', 'ChEBI identifier']]
        .rename(columns={'Reaction identifier':'source_id','ChEBI identifier':'target_id'})
        .assign(source_layer='rhea_reactionid', target_layer='rhea_chebiid', interlayer=False)
        .drop_duplicates()
    )

    # Reaction ↔ EC edges (reuse exploded enzyme-class rows)
    df_rhea_exploded_ec = explode_columns(df_rhea, ['EC number'], delimiter=';')
    df_edges_reaction_ec = (
        df_rhea_exploded_ec[['Reaction identifier', 'EC number']]
        .rename(columns={'Reaction identifier':'source_id','EC number':'target_id'})
        .assign(source_layer='rhea_reactionid', target_layer='rhea_ec', interlayer=False)
        .drop_duplicates()
    )

    # Combine edges
    # df_edges = pd.concat([df_edges_ec, df_edges_reaction_chebi, df_edges_reaction_ec], ignore_index=True).drop_duplicates()
    df_edges_ec = df_edges_ec.assign(edge_type="ec_hierarchy")
    df_edges_reaction_chebi = df_edges_reaction_chebi.assign(edge_type="reaction_chebi")
    df_edges_reaction_ec = df_edges_reaction_ec.assign(edge_type="reaction_ec")

    df_edges = pd.concat(
        [df_edges_ec, df_edges_reaction_chebi, df_edges_reaction_ec],
        ignore_index=True
    ).drop_duplicates()
    if verbose:
        print(f"Combined edges: {df_edges.shape[0]} unique edges")

    # Build node DataFrame manually to preserve metadata
    # Reaction nodes with full metadata
    df_nodes_reaction = df_rhea[[
        'Reaction identifier', 'Equation', 'ChEBI identifier', 'ChEBI name',
        'EC number', 'Enzymes', #note: dropped 'Participant identifier' and 'Enzyme class' bc not part of REST API (seems to be post-processed on client side), not immediately useful, could add downstream
        'Gene Ontology', 'Cross-reference (Reactome)'
    ]].rename(columns={'Reaction identifier':'node_id','ChEBI name':'chebi_name'}) \
      .assign(layer='rhea_reactionid') \
      .drop_duplicates()

    # ChEBI nodes
    df_nodes_chebi = df_rhea_exploded[['ChEBI identifier', 'ChEBI name']] \
        .rename(columns={'ChEBI identifier':'node_id','ChEBI name':'chebi_name'}) \
        .assign(layer='rhea_chebiid') \
        .drop_duplicates()

    # EC nodes (from earlier helper)
    # df_nodes_ec already has 'node_id', 'layer', 'ec_level'
  
    # Combine all node types into a single DataFrame
    df_nodes = pd.concat([df_nodes_reaction, df_nodes_chebi, df_nodes_ec], ignore_index=True, sort=False)
    df_nodes = clean_missing_strings(df_nodes).drop_duplicates()
    if verbose:
        print(f"Built nodes: {df_nodes.shape[0]} nodes, {df_nodes.shape[1]} columns")

    result = {'df_edges': df_edges, 'df_nodes': df_nodes}

    # ---- write processed cache ----
    if use_cache:
        if verbose:
            print("↪ Caching Rhea (processed) nodes & edges")
        save_cache("rhea", df_nodes=result["df_nodes"], df_edges=result["df_edges"])

    return result

def main():
    parser = argparse.ArgumentParser(description="Process Rhea data using LipiNet")
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--use-cache', action='store_true', help='Load/save processed nodes & edges cache')
    parser.add_argument('--force-download', action='store_true', help='Force fresh raw download and rebuild')
    args = parser.parse_args()
    
    verbose = not args.quiet
    results = parse_rhea_data(verbose=verbose, use_cache=args.use_cache, force_download=args.force_download)
    if verbose:
        print("Rhea processing complete. DataFrames are ready.")

if __name__ == "__main__":
    main()
