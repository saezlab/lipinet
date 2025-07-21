import os
import requests
import pandas as pd
import gzip
import io
import json
import re

def download_and_load_data(filename, url, file_format='csv', compressed=False, sep=',', encoding='utf-8', verbose=False):
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

    Returns:
    - data (DataFrame, dict, or list): The loaded data from the file, in the format specified.
    """
    # Set the directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '.data/downloaded')
    os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists

    # Define the full path to the file
    filepath = os.path.join(data_dir, filename)

    # Check if the file already exists
    if not os.path.exists(filepath):
        if verbose:
            print(f"File not found locally. Downloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raises an error if the download fails
        
        # Handle gzip-compressed files
        if compressed:
            with gzip.open(io.BytesIO(response.content), 'rt', encoding=encoding) as f:
                if file_format == 'csv' or file_format == 'tsv':
                    data = pd.read_csv(f, sep=sep, low_memory=False)
                else:
                    raise ValueError("Unsupported file format with gzip compression. Only 'csv' is supported.")
        else:
            # Save uncompressed content to local file
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Load uncompressed content
            if file_format == 'csv' or file_format == 'tsv':
                data = pd.read_csv(filepath, sep=sep, low_memory=False) #, encoding=encoding)
            elif file_format == 'json':
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Only 'json', 'csv' and 'tsv' are supported.")

        if verbose:
            print(f"Data downloaded and saved to {filepath}.")
    else:
        if verbose:
            print(f"File found locally at {filepath}. Loading data...")

        # Load the file from local storage
        if file_format == 'csv' or file_format == 'tsv':
            data = pd.read_csv(filepath, sep=sep, low_memory=False) #, encoding=encoding)
        elif file_format == 'json':
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Only 'json', 'csv' and 'tsv' are supported.")
    
    # Save decompressed data locally if it was downloaded as gzip
    if compressed and not os.path.exists(filepath):
        data.to_csv(filepath, sep=sep, index=False) #, encoding=encoding) # if the original csv/tsv was encoded, we won't worry about that when we save it
    
    return data



def get_prior_knowledge(name_of_resource, verbose=False):
    #note these will be added to the data dir (.data/databases)
    resources = {
        'swisslipids':
            {'filename': 'swisslipids_lipids.tsv', 
            'data_url': "https://www.swisslipids.org/api/file.php?cas=download_files&file=lipids.tsv"},
        'rhea': 
            {'filename': 'rhea.tsv',
            'data_url': 'https://www.rhea-db.org/rhea/?query=&columns=rhea-id,equation,chebi,chebi-id,ec,uniprot,go,pubmed,reaction-xref(EcoCyc),reaction-xref(MetaCyc),reaction-xref(KEGG),reaction-xref(Reactome),reaction-xref(M-CSA)&format=tsv&limit=1000000'}
    }

    try: 
        local_filename = resources[name_of_resource]['filename']
        data_url = resources[name_of_resource]['data_url']
        if name_of_resource=='swisslipids':
            fetched_data = download_and_load_data(local_filename, data_url, file_format='tsv', compressed=True, sep='\t', encoding='latin-1', verbose=verbose)
            fetched_data = clean(fetched_data, name_of_resource=name_of_resource, verbose=verbose)
        else:
            fetched_data = download_and_load_data(local_filename, data_url, file_format='tsv', sep='\t', verbose=verbose)
        return fetched_data
    except KeyError as e:
        raise e(f"KeyError encountered, probably because the resource you requested is not yet supported.")
    

def clean(df, name_of_resource, verbose=False):
    """
    Some of the data sources need specialised cleaning to make them nicer to work with.
    """    
    if name_of_resource=='swisslipids':
        # Note the swisslipids 'Lipid class*' column has some strings ending with an empty space, which can really screw with the hierarchy...
        if verbose:
            print("Before cleaning, number of values in lipid class column with trailing space:", df["Lipid class*"].str.endswith(" ").value_counts())
        #df['Lipid class*'] = df['Lipid class*'].str.strip(' ')
        df = clean_columns(df, cols=['Lipid class*','CHEBI'], strip_chars=' ', verbose=verbose)
        df['CHEBI'] = df['CHEBI'].replace('CHEBI:', '', regex=True)  # remove 'CHEBI:' prefix
        df['CHEBI'] = df['CHEBI'].replace(' ', '', regex=True)

        if verbose:
            print("After cleaning, number of values in lipid class column with trailing space:", df["Lipid class*"].str.endswith(" ").value_counts())
        return df
    

def clean_columns(
    df: pd.DataFrame,
    cols: list[str],
    strip_chars: str | None = None,
    trim_substrings: list[str] | None = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    For each col in `cols`:
      1. .str.strip(strip_chars)  — if strip_chars is None defaults to whitespace
      2. remove any of the `trim_substrings` at start or end
    
    Args:
        df:              your DataFrame
        cols:            list of column names to clean (if empty, all columns)
        strip_chars:     string of characters to strip from ends (None → whitespace)
        trim_substrings: list of literal substrings to drop if they appear at start or end
        verbose:         print before/after samples
    """
    if len(cols) == 0:
        cols = df.columns.tolist()

    for col in cols:
        if verbose:
            print(f"\n>> Cleaning “{col}”:")
            print("   sample before:", df[col].astype(str).head().tolist())
        # ensure strings
        s = df[col].astype(str)

        # 1) strip characters (whitespace if strip_chars is None)
        s = s.str.strip(strip_chars)

        # 2) trim any of the given substrings from either end
        if trim_substrings:
            # build a regex like '^(?:sub1|sub2)+|(?:sub1|sub2)+$'
            esc = [re.escape(x) for x in trim_substrings]
            pat = rf'^(?:{"|".join(esc)})+|(?:{"|".join(esc)})+$'
            s = s.str.replace(pat, "", regex=True)

        df[col] = s
        if verbose:
            print("   sample after: ", df[col].head().tolist())
    return df.copy()