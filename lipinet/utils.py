import numpy as np
import pandas as pd

def split_and_expand_large(df, split_col, delimiter, expand_cols):
    """
    Splits a column by a delimiter and expands specified columns for large DataFrames, handling None/NaN values.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    split_col (str): The name of the column to split.
    delimiter (str): The delimiter to split the column by.
    expand_cols (list): List of column names to be expanded with the split column.

    Returns:
    pd.DataFrame: A new DataFrame with the split and expanded rows.
    """
    # Step 1: Split the split_col into lists, handling None/NaN as empty lists
    split_data = df[split_col].apply(lambda x: str(x).split(delimiter) if pd.notnull(x) else [np.nan])
    
    # Step 2: Calculate the number of splits for each row to repeat other columns
    repeat_counts = split_data.apply(len)
    
    # Step 3: Create a DataFrame with repeated values for expand_cols
    expanded_data = {col: np.repeat(df[col].values, repeat_counts) for col in expand_cols}
    
    # Step 4: Flatten the split_data and assign to the expanded split_col
    expanded_data[split_col] = np.concatenate(split_data.values)
    
    # Step 5: Create the expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data)
    
    return expanded_df

    # # Example usage for large DataFrames with None or NaN values
    # data = {'col1': ['word|smith', None, 'apple|banana|cherry', np.nan], 'col2': ['john', 'doe', 'alice', 'bob']}
    # df = pd.DataFrame(data)
    # result = split_and_expand_large(df, split_col='col1', delimiter='|', expand_cols=['col2'])
    # print(result)

def create_nodedf_from_edgedf(edge_df, props=['layer', 'id'], cols=['layer', 'node_id']):
    dfs = []
    for prop in props:
        df = pd.melt(edge_df, 
                    value_vars=[f'source_{prop}',f'target_{prop}'], 
                    var_name='type', value_name='value')
        dfs.append(df)
    node_df = pd.concat(dfs, axis=1)
    node_df = node_df['value']
    node_df.columns = cols
    return node_df