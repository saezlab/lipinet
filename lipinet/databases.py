import os
import requests
import pandas as pd
import gzip
import io
import json

def download_and_load_data(filename, url, file_format='csv', compressed=False, sep=',', encoding='utf-8'):
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

        print(f"Data downloaded and saved to {filepath}.")
    else:
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



def get_prior_knowledge(name_of_resource):
    resources = {
        'swisslipids':{'filename': 'swisslipids_lipids.tsv', #note this will be joined to the data dir (.data/databases)
                       'data_url': "https://www.swisslipids.org/api/file.php?cas=download_files&file=lipids.tsv"}
    }

    try: 
        local_filename = resources[name_of_resource]['filename']
        data_url = resources[name_of_resource]['data_url']
        if name_of_resource=='swisslipids':
            fetched_data = download_and_load_data(local_filename, data_url, file_format='tsv', compressed=True, sep='\t', encoding='latin-1')
            fetched_data = clean(fetched_data, name_of_resource=name_of_resource)
        else:
            fetched_data = download_and_load_data(local_filename, data_url, file_format='tsv', sep='\t')
        return fetched_data
    except KeyError as e:
        raise e(f"KeyError encountered, probably because the resource you requested is not yet supported.")
    

def clean(df, name_of_resource):
    """
    Some of the data sources need specialised cleaning to make them nicer to work with.
    """

    if name_of_resource=='swisslipids':
        # Note the swisslipids 'Lipid class*' column has some strings ending with an empty space, which can really screw with the hierarchy...
        print(f'Before cleaning, number of values in lipid class column with trailing space: {df['Lipid class*'].str.endswith(' ').value_counts()}')
        df['Lipid class*'] = df['Lipid class*'].str.strip(' ')
        print(f'After cleaning, number of values in lipid class column with trailing space: {df['Lipid class*'].str.endswith(' ').value_counts()}')
        return df