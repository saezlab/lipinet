import os
import requests
import pandas as pd
import gzip
import io
import json
import re

from lipinet.utils import clean_missing_strings, clean_columns

def download_and_load_data(filename, url, file_format='csv', compressed=False, sep=',', encoding='utf-8', header='infer', verbose=False, force_download=False):
    """
    Checks if the specified file exists locally. If not, downloads it from the provided URL.
    Supports loading compressed files and handling different formats.

    Parameters:
    - filename (str): The name of the file to be saved within the data directory.
    - url (str): The URL to download the file from if it's not found locally.
    - file_format (str): The format of the file ('json' or 'csv'). Defaults to 'csv'.
    - compressed (bool): If True, expects the downloaded file to be in gzip format. Defaults to False.
    - sep (str): Separator to use if loading CSV/TSV data. Defaults to ','.
    - encoding (str): Encoding to use for reading files. Defaults to 'utf-8'.
    - verbose (bool): If True, prints additional information during the process. Defaults to False.
    - force_download (bool): If True, download even if the file exists locally. Defaults to False.
    - header: passed to pandas.read_csv. Use None for no header, 0 or 'infer' for first-row header.

    Returns:
    - data (DataFrame, dict, or list): The loaded data from the file, in the format specified.
    """
    # Set the directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '.data/downloaded')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)

    need_download = force_download or not os.path.exists(filepath)

    if need_download:
        if verbose:
            if force_download and os.path.exists(filepath):
                print(f"Override requested; re-downloading {filename} from {url} even though it exists locally.")
            else:
                print(f"File not found locally. Downloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        if compressed:
            # Decompress in-memory and load
            with gzip.open(io.BytesIO(response.content), 'rt', encoding=encoding) as f:
                if file_format in ('csv', 'tsv'):
                    data = pd.read_csv(f, sep=sep, low_memory=False, header=header)
                else:
                    raise ValueError("Unsupported file format with gzip compression. Only 'csv'/'tsv' are supported.")
            # Always overwrite the decompressed file on disk
            data.to_csv(filepath, sep=sep, index=False)
        else:
            # Save raw content
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Load it
            if file_format in ('csv', 'tsv'):
                data = pd.read_csv(filepath, sep=sep, low_memory=False, header=header)
            elif file_format == 'json':
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Only 'json', 'csv', and 'tsv' are supported.")

        if verbose:
            print(f"Data downloaded and saved to {filepath}.")
    else:
        if verbose:
            print(f"File found locally at {filepath}. Loading data...")
        if file_format in ('csv', 'tsv'):
            data = pd.read_csv(filepath, sep=sep, low_memory=False, header=header)
        elif file_format == 'json':
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Only 'json', 'csv', and 'tsv' are supported.")

    return data



def get_prior_knowledge(name_of_resource, verbose=False, force_download=False, squeeze=True):
    #note these will be added to the data dir (.data/databases)
    resources = {
        'swisslipids':
            [{'filename': 'swisslipids_lipids.tsv', 
              'data_url': "https://www.swisslipids.org/api/file.php?cas=download_files&file=lipids.tsv"}],
        'rhea': 
            [{'filename': 'rhea.tsv',
              'data_url': 'https://www.rhea-db.org/rhea/?query=&columns=rhea-id,equation,chebi,chebi-id,ec,uniprot,go,pubmed,reaction-xref(EcoCyc),reaction-xref(MetaCyc),reaction-xref(KEGG),reaction-xref(Reactome),reaction-xref(M-CSA)&format=tsv&limit=1000000'}],
        'reactome':
            [{'filename': 'ChEBI2Reactome_PE_All_Levels.tsv',
              'data_url': 'https://reactome.org/download/current/ChEBI2Reactome_PE_All_Levels.txt'},
            {'filename': 'ChEBI2Reactome_PE_Reactions.tsv',
              'data_url': 'https://reactome.org/download/current/ChEBI2Reactome_PE_Reactions.txt'},
            {'filename': 'ReactomePathways.tsv',
              'data_url': 'https://reactome.org/download/current/ReactomePathways.txt'},
            {'filename': 'ReactomePathwaysRelation.tsv',
              'data_url': 'https://reactome.org/download/current/ReactomePathwaysRelation.txt'},
            ]
    }

    try:
        fetched_data = {}
        for file in resources[name_of_resource]: 
            local_filename = file['filename']
            data_url = file['data_url']
            if verbose: 
                print(f'Fetching {local_filename}')
            if name_of_resource=='swisslipids':
                fetched_data_intermediate = download_and_load_data(local_filename, data_url, file_format='tsv', compressed=True, sep='\t', encoding='latin-1', verbose=verbose, force_download=force_download)
                fetched_data_intermediate = clean(fetched_data_intermediate, name_of_resource=name_of_resource, verbose=verbose)
            elif name_of_resource=='reactome':
                fetched_data_intermediate = download_and_load_data(local_filename, data_url, file_format='tsv', sep='\t', header=None, verbose=verbose, force_download=force_download)
                fetched_data_intermediate = clean(fetched_data_intermediate, name_of_resource=name_of_resource, verbose=verbose, filename=local_filename)
            else:
                fetched_data_intermediate = download_and_load_data(local_filename, data_url, file_format='tsv', sep='\t', verbose=verbose, force_download=force_download)
            fetched_data[local_filename] = fetched_data_intermediate
        # if only 1 df in fetched data and squeeze==True, will return just that df, else will return a dictonary with all dfs
        if squeeze and len(fetched_data)==1:
            (key, value), = fetched_data.items()
            if verbose: 
                print(f'Returning {key} as a single df')
            return value
        else: 
            if verbose: 
                print(f'Returning {[key for key in fetched_data.keys()]} as a dict of dfs')
            return fetched_data
    except KeyError as e:
        raise KeyError(
            f"Unknown resource {name_of_resource!r} or malformed resource entry."
        ) from e
    

def clean(df: pd.DataFrame, name_of_resource: str, verbose: bool = False, filename: str = None) -> pd.DataFrame:
    """
    Dispatch per-resource specialized cleaning.
    Returns a cleaned copy; original df is not mutated.
    """
    df = df.copy()
    if name_of_resource == 'swisslipids':
        if verbose:
            trailing_before = df["Lipid class*"].str.endswith(" ").value_counts(dropna=False)
            print("Before cleaning, trailing-space counts in 'Lipid class*':", trailing_before.to_dict())

        # General whitespace stripping and removal of 'CHEBI:' prefix from CHEBI
        df = clean_columns(
            df,
            cols=['Lipid class*', 'CHEBI'],
            strip_chars=' ',
            trim_substrings=['CHEBI:'],  # removes CHEBI: from ends (mostly prefix), only the case for a very few rows with CHEBI present
            collapse_whitespace=True,
            verbose=verbose,
            ignore_missing=False  # fail early if expected column missing
        )

        # Additional CHEBI-specific normalization: remove internal spaces if any (e.g., " 12345 ")
        if 'CHEBI' in df.columns:
            df.loc[:, 'CHEBI'] = df['CHEBI'].astype("string").str.replace(r'\s+', '', regex=True)
            # A very small number of rows have random CHEBI ids in their middle (e.g. 82731|CHEBI:82731). We want to handle this.
            df.loc[:, 'CHEBI'] = df['CHEBI'].astype("string").str.replace('CHEBI:', '')

        if verbose:
            trailing_after = df["Lipid class*"].str.endswith(" ").value_counts(dropna=False)
            print("After cleaning, trailing-space counts in 'Lipid class*':", trailing_after.to_dict())

        return df
    if name_of_resource == 'reactome':
        # Reactome does not currently include header column names for the files we downloaded. So we will add this annotation ourselves.
        pe_dict = {
            0: 'source_db_identifier',
            1: 'reactome_pe_stableid',
            2: 'reactome_pe_name',
            3: 'reactome_pathway_stableid',
            4: 'url',
            5: 'event_name_pathway_or_reaction',
            6: 'evidence_code',
            7: 'species'
        }
        path_dict = {
            0: 'reactome_pathway_stableid',
            1: 'reactome_pathway_name',
            2: 'species'
        }
        path_dict_rel = {
            0: 'parent_stableid',
            1: 'child_stableid',
        }
        print(f'CLEANING REACTOME!, filename: {filename}')
        if filename in ['ChEBI2Reactome_PE_All_Levels.tsv', 'ChEBI2Reactome_PE_Reactions.tsv']:
            print(df)
            df = df.rename(columns=pe_dict)
        elif filename in ['ReactomePathways.tsv']:
            df = df.rename(columns=path_dict)
        elif filename in ['ReactomePathwaysRelation.tsv']:
            df = df.rename(columns=path_dict_rel)
        else:
            pass
        print(df)
        return df

    # fallback: no resource-specific rules, return copy with optional notice
    if verbose:
        print(f"No specialized cleaning defined for resource '{name_of_resource}'; returning original dataframe copy.")
    return df
