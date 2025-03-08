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
    
    # Add a parsed version of the Components column
    df_swisslipids['Components_parsed'] = df_swisslipids['Components*']
    
    if verbose:
        print("Initial SwissLipids DataFrame (first 5 rows):")
        print(df_swisslipids.head(), "\n")
    
    # Melt the dataframe to create an edges dataframe
    df_swisslipids_edges = pd.melt(
        df_swisslipids,
        id_vars=['Lipid ID'],
        value_vars=[
            'CHEBI', 'LIPID MAPS', 'HMDB', 'MetaNetX', 'PMID',
            'Lipid class*', 'Abbreviation*', 'Synonyms*', 'Parent',
            'Components*', 'Components_parsed'
        ],
        var_name='melted_column',
        value_name='value'
    )
    df_swisslipids_edges = df_swisslipids_edges.dropna(subset=['value'])
    
    # Set static source layer and rename columns
    df_swisslipids_edges['source_layer'] = 'swisslipids'
    df_swisslipids_edges.rename(
        columns={
            'Lipid ID': 'source_id',
            'melted_column': 'target_layer',
            'value': 'target_id'
        },
        inplace=True
    )
    df_swisslipids_edges = df_swisslipids_edges[['source_layer', 'source_id', 'target_layer', 'target_id']]
    
    # Update target_layer: if 'Lipid class*' then set to 'swisslipids', otherwise format the string
    df_swisslipids_edges['target_layer'] = df_swisslipids_edges['target_layer'].map(
        lambda x: 'swisslipids' if x=='Lipid class*' else f"sl_{str(x).replace(' ','').strip('*').lower()}"
    )
    
    # For rows where both source and target layers are 'swisslipids', swap the columns so that the parent points to the children
    condition = (
        (df_swisslipids_edges["source_layer"] == "swisslipids") &
        (df_swisslipids_edges["target_layer"] == "swisslipids")
    )
    df_swisslipids_edges.loc[condition, ["source_layer", "source_id", "target_layer", "target_id"]] = \
        df_swisslipids_edges.loc[condition, ["target_layer", "target_id", "source_layer", "source_id"]].values
    
    # Handle multilinks: edges with '|' in target_id
    edges_with_multilinks = df_swisslipids_edges[
        df_swisslipids_edges['target_id'].str.contains('|', regex=False, na=False)
    ]
    edges_with_multilinks_split = split_and_expand_large(
        edges_with_multilinks, 
        split_col='target_id', 
        expand_cols=['source_layer', 'source_id', 'target_layer'],
        delimiter='|'
    ).drop_duplicates()
    
    # Handle multilinks for components: edges with '/' in target_id and target_layer contains 'sl_components'
    edges_with_multilinks2 = df_swisslipids_edges[
        df_swisslipids_edges['target_id'].str.contains('/', regex=False, na=False) &
        df_swisslipids_edges['target_layer'].str.contains('sl_components', regex=False, na=False)
    ]
    edges_with_multilinks2_split = split_and_expand_large(
        edges_with_multilinks2, 
        split_col='target_id', 
        expand_cols=['source_layer', 'source_id', 'target_layer'],
        delimiter='/'
    ).drop_duplicates()
    
    # For parsed components, remove any parenthesized info (e.g., '(sn2)')
    mask = edges_with_multilinks2_split['target_layer'] == 'sl_components_parsed'
    edges_with_multilinks2_split.loc[mask, 'target_id'] = \
        edges_with_multilinks2_split.loc[mask, 'target_id'].str.split('(').str[0].str.strip()
    
    # Remove the original problematic multilink rows and combine with the corrected ones
    mask_pipe = df_swisslipids_edges['target_id'].str.contains('|', regex=False, na=False)
    mask_slash = (
        df_swisslipids_edges['target_id'].str.contains('/', regex=False, na=False) &
        df_swisslipids_edges['target_layer'].str.contains('sl_components', regex=False, na=False)
    )
    mask_problem = mask_pipe | mask_slash
    df_clean = df_swisslipids_edges[~mask_problem].copy()
    df_swisslipids_edges = pd.concat([
        df_clean, 
        edges_with_multilinks_split, 
        edges_with_multilinks2_split
    ], ignore_index=True)
    df_swisslipids_edges = df_swisslipids_edges.drop_duplicates()
    
    # Add an 'interlayer' column indicating whether the edge is between different layers
    df_swisslipids_edges['interlayer'] = df_swisslipids_edges['source_layer'] != df_swisslipids_edges['target_layer']
    
    if verbose:
        print("Processed edges DataFrame (first 5 rows):")
        print(df_swisslipids_edges.head(), "\n")
    
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
    
    # Merge node information with additional details from the original SwissLipids dataframe
    df_swisslipids_nodes = pd.merge(
        df_swisslipids_nodes, 
        df_swisslipids.assign(from_layer_col='swisslipids'),
        left_on=['layer', 'node_id'], 
        right_on=['from_layer_col', 'Lipid ID'],
        how='outer'
    )
    
    # Clean up duplicates and remove the auxiliary 'from_layer_col'
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
