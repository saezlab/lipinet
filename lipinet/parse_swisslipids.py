#!/usr/bin/env python3
"""
A standalone module that loads and processes SwissLipids data into a df_nodes using lipinet.

This module provides a helper function `parse_swisslipids_data` that can be imported into notebooks or other scripts.
A thin wrapper in the `main()` function allows command-line execution.
"""

import argparse
import importlib
import pandas as pd

# Import lipinet module and related functions
import lipinet.databases  # Import the module

# Reload the module to ensure changes are picked up
importlib.reload(lipinet)

from lipinet.databases import get_prior_knowledge
from lipinet.utils import split_and_expand_large, create_nodedf_from_edgedf


def parse_swisslipids_data(verbose=False):
    """Core function to process SwissLipids data and return nodes and edges dataframes.

    Parameters:
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        dict: A dictionary with keys 'df_nodes' and 'df_edges'.
    """
    # Load the SwissLipids data and add a layer column
    df_swisslipids = get_prior_knowledge('swisslipids')
    df_swisslipids['from_layer_col'] = 'swisslipids'
    if verbose:
        print("Initial SwissLipids DataFrame (first 5 rows):")
        print(df_swisslipids.head(), "\n")
    
    # Check for double entries in the 'Lipid class*' column
    double_entries = df_swisslipids.dropna(subset=['Lipid class*'])[df_swisslipids['Lipid class*'].dropna().str.contains('|', regex=False)]
    if verbose:
        print("Double entries in 'Lipid class*':")
        print(double_entries, "\n")
    
    # Split the 'Lipid class*' column into multiple rows
    df_swisslipids_splitexp = split_and_expand_large(
        df_swisslipids, 
        split_col='Lipid class*', 
        expand_cols=['Lipid ID', 'Level', 'Name', 'Abbreviation*',
                     'CHEBI', 'LIPID MAPS', 'HMDB', 'MetaNetX', 'PMID', 'from_layer_col'],
        delimiter='|'
    )
    
    # Melt the expanded dataframe to create an edges dataframe
    df_swisslipids_edges = pd.melt(
        df_swisslipids_splitexp, 
        id_vars=['Lipid ID'], 
        value_vars=['CHEBI', 'LIPID MAPS', 'HMDB', 'MetaNetX', 'PMID', 'Lipid class*'], 
        var_name='melted_column', 
        value_name='value'
    )
    
    # Prepare the edges dataframe
    df_swisslipids_edges['source_layer'] = 'swisslipids'
    df_swisslipids_edges.rename(
        columns={'Lipid ID': 'source_id', 'melted_column': 'target_layer', 'value': 'target_id'}, 
        inplace=True
    )
    df_swisslipids_edges = df_swisslipids_edges[['source_layer', 'source_id', 'target_layer', 'target_id']]
    df_swisslipids_edges['target_layer'] = df_swisslipids_edges['target_layer'].map(
        lambda x: 'swisslipids' if x == 'Lipid class*' else x
    )
    df_swisslipids_edges['target_layer'] = df_swisslipids_edges['target_layer'].map(
        lambda x: str(x).replace(' ', '').strip('*').lower()
    )
    
    # For rows where both source_layer and target_layer are 'swisslipids', swap columns so that the parent points to the children
    condition = (
        (df_swisslipids_edges["source_layer"] == "swisslipids") &
        (df_swisslipids_edges["target_layer"] == "swisslipids")
    )
    df_swisslipids_edges.loc[condition, ["source_layer", "source_id", "target_layer", "target_id"]] = \
        df_swisslipids_edges.loc[condition, ["target_layer", "target_id", "source_layer", "source_id"]].values
    
    # Define function to assess if an edge is interlayer or intralayer
    def assess_edge_layertype(df):
        df['interlayer'] = df['source_layer'] != df['target_layer']
        return df 
    
    df_swisslipids_edges = assess_edge_layertype(df_swisslipids_edges)
    if verbose:
        print("Processed edges DataFrame (first 5 rows):")
        print(df_swisslipids_edges.head(), "\n")
    
    # Group and count edges by target_layer and target_id (for diagnostic purposes)
    edge_counts = df_swisslipids_edges.groupby('target_layer').value_counts(subset=['target_id'], dropna=False)
    if verbose:
        print("Edge counts by target_layer and target_id:")
        print(edge_counts, "\n")
    
    # Create the node dataframe from the edges dataframe
    df_swisslipids_nodes = create_nodedf_from_edgedf(
        edge_df=df_swisslipids_edges, 
        props=['layer', 'id'], 
        cols=['layer', 'node_id']
    )
    if verbose:
        print("Initial node DataFrame (first 5 rows):")
        print(df_swisslipids_nodes.head(), "\n")
        print("Duplicate counts in node DataFrame:")
        print(df_swisslipids_nodes.value_counts(dropna=True), "\n")
    
    # Merge node information with additional details from the expanded SwissLipids dataframe
    df_swisslipids_nodes = pd.merge(
        df_swisslipids_nodes, 
        df_swisslipids_splitexp,
        left_on=['layer', 'node_id'], 
        right_on=['from_layer_col', 'Lipid ID'],
        how='outer'
    )
    
    # Remove duplicates and drop the 'from_layer_col' column
    df_swisslipids_nodes = df_swisslipids_nodes.drop_duplicates()
    if 'from_layer_col' in df_swisslipids_nodes.columns:
        df_swisslipids_nodes = df_swisslipids_nodes.drop(columns='from_layer_col')
    if verbose:
        print("Final node DataFrame (first 5 rows):")
        print(df_swisslipids_nodes.head())
    
    return {"df_nodes": df_swisslipids_nodes, "df_edges": df_swisslipids_edges}


def main():
    """Thin wrapper for command-line execution."""
    parser = argparse.ArgumentParser(description="Process SwissLipids data using lipinet")
    parser.add_argument('--quiet', action='store_true', help='Run in quiet mode (minimal output)')
    args = parser.parse_args()
    
    # Set verbose flag based on the --quiet argument
    verbose = not args.quiet
    
    results = parse_swisslipids_data(verbose=verbose)
    if verbose:
        print("\nProcessing complete. The data has been parsed into nodes and edges.")


if __name__ == '__main__':
    main()
