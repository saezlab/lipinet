from collections.abc import Iterable
from pathlib import Path
import re

from IPython.display import display
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

CACHE_ROOT = Path(__file__).parent / ".data" / "processed"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def split_and_expand_large(df, split_col, delimiter, expand_cols):
    """
    Splits a column by a delimiter and expands specified columns for large DataFrames, handling None/NaN values.

    Parameters
    ----------
    df (pd.DataFrame): The original DataFrame.
    split_col (str): The name of the column to split.
    delimiter (str): The delimiter to split the column by.
    expand_cols (list): List of column names to be expanded with the split column.

    Returns
    -------
    pd.DataFrame: A new DataFrame with the split and expanded rows.
    """
    # Edge case: empty input
    if df is None or len(df) == 0:
        # Return an empty frame with expected columns
        return pd.DataFrame(
            {
                **{
                    c: pd.Series(dtype=df[c].dtype if c in df.columns else object)
                    for c in expand_cols
                },
                split_col: pd.Series(dtype=object),
            },
        )

    # Step 1: Split the split_col into lists, handling None/NaN as empty lists
    split_data = df[split_col].apply(
        lambda x: str(x).split(delimiter) if pd.notnull(x) else [np.nan],
    )

    # Step 2: Calculate the number of splits for each row to repeat other columns
    repeat_counts = split_data.apply(len)

    # Step 3: Create a DataFrame with repeated values for expand_cols
    expanded_data = {col: np.repeat(df[col].values, repeat_counts) for col in expand_cols}

    # Step 4: Flatten the split_data and assign to the expanded split_col
    # If there are no splits (shouldn't happen due to early return), guard
    if len(split_data.values) == 0:
        return pd.DataFrame(
            {
                **{
                    c: pd.Series(dtype=df[c].dtype if c in df.columns else object)
                    for c in expand_cols
                },
                split_col: pd.Series(dtype=object),
            },
        )
    expanded_data[split_col] = np.concatenate(split_data.values)

    # Step 5: Create the expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data)

    return expanded_df

    # # Example usage for large DataFrames with None or NaN values
    # data = {'col1': ['word|smith', None, 'apple|banana|cherry', np.nan], 'col2': ['john', 'doe', 'alice', 'bob']}
    # df = pd.DataFrame(data)
    # result = split_and_expand_large(df, split_col='col1', delimiter='|', expand_cols=['col2'])
    # print(result)


def create_nodedf_from_edgedf(edge_df, props=None, cols=None):
    """
    Create a node DataFrame from an edge DataFrame by stacking source/target
    columns for the given properties.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Edge dataframe with columns like 'source_layer','source_id',
        'target_layer','target_id'.
    props : list[str]
        Two-element list specifying the property suffixes to pull from
        the edge dataframe (default ['layer','id']).
    cols : list[str]
        Output column names for the resulting node dataframe
        (default ['layer','node_id']).

    Returns
    -------
    pd.DataFrame
        Unique nodes with columns named per `cols`.
    """
    if props is None:
        props = ["layer", "id"]
    if cols is None:
        cols = ["layer", "node_id"]

    if len(props) != 2 or len(cols) != 2:
        raise ValueError("props and cols must be length-2 lists")

    src = edge_df[[f"source_{props[0]}", f"source_{props[1]}"]].rename(
        columns={f"source_{props[0]}": cols[0], f"source_{props[1]}": cols[1]},
    )
    tgt = edge_df[[f"target_{props[0]}", f"target_{props[1]}"]].rename(
        columns={f"target_{props[0]}": cols[0], f"target_{props[1]}": cols[1]},
    )
    node_df = pd.concat([src, tgt], ignore_index=True).drop_duplicates()
    return node_df


def check_for_split_characters(df, delimiter="|"):
    cols_with_split_chars = []
    for col in df.columns:
        print(f"Checking split characters ({delimiter}) in " + col)
        try:
            temp = df[df[col].str.contains(delimiter, regex=False, na=False)]
            if len(temp) > 0:
                print(f"Found {len(temp)} rows with split characters")
                display(temp)
                cols_with_split_chars.append(col)
            else:
                print("No rows found\n")
        except AttributeError:
            print("Not a string column\n")
    return cols_with_split_chars


def clean_missing_strings(
    df: pd.DataFrame,
    cols=None,
    string_fraction_threshold=0.9,
) -> pd.DataFrame:
    """
    Strip whitespace from stringy values and normalize common placeholder "missing" strings
    into real pandas NA. Operates on specified columns or all object/string columns by default.

    Args:
        df: input DataFrame (modified in-place).
        cols: list of columns to process; if None, uses all object/string dtype columns.
        string_fraction_threshold: for object dtype columns, if >= this fraction of non-null
            values are str, coerce to StringDtype and vectorize the strip; otherwise do per-element.
    """
    if cols is None:
        cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # Preserve original values for non-string entries so we never coerce them away
    _orig = {c: df[c].copy() for c in cols if c in df.columns}
    for c in cols:
        if c not in df.columns:
            continue  # or warn

        series = df[c]
        # Fully vectorized path for dedicated string dtype
        if ptypes.is_string_dtype(series.dtype):
            df[c] = series.str.strip()
        elif ptypes.is_object_dtype(series.dtype):
            nonnull = series.dropna()
            if len(nonnull) == 0:
                continue  # nothing to do
            # Heuristic: are most values strings? If so, cast to StringDtype and vectorize.
            is_str_mask = nonnull.map(lambda x: isinstance(x, str))
            frac = is_str_mask.mean()
            if frac >= string_fraction_threshold:
                # Safe to coerce: convert entire column to StringDtype, strip vectorially
                s = series.astype("string")
                df[c] = s.str.strip()
            else:
                # Mixed types: only strip actual python str values
                cleaned = series.copy()
                mask = series.map(lambda x: isinstance(x, str))
                cleaned.loc[mask] = cleaned.loc[mask].str.strip()
                df[c] = cleaned
        else:
            # non-string column: leave alone
            continue

    # Global placeholder normalization on string-like columns only.
    # Use pandas StringDtype and .str.replace to avoid numpy vectorize issues
    # on empty object blocks.
    for col in df.columns:
        s = df[col]
        if ptypes.is_string_dtype(s.dtype) or ptypes.is_object_dtype(s.dtype):
            # Only consider original string entries for normalization
            is_str = s.map(lambda x: isinstance(x, str))
            if is_str.any():
                s_str = s.astype("string")
                mask_placeholder = s_str.str.match(r"(?i)^(nan|none|null)$", na=False)
                mask_empty = s_str.str.fullmatch(r"\s*", na=False)
                apply_mask = is_str & (mask_placeholder | mask_empty)
                res = s.copy()
                # Use Python None to avoid coercing non-strings; tests treat None as missing
                res.loc[apply_mask] = None
                df[col] = res

    # Restore any non-string entries from the original to ensure they remain unchanged
    for c, orig in _orig.items():
        if c in df.columns:
            non_str_mask = orig.map(lambda x: not isinstance(x, str))
            if non_str_mask.any():
                df.loc[non_str_mask, c] = orig.loc[non_str_mask]
    return df


def clean_columns(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    strip_chars: str | None = None,
    trim_substrings: Iterable[str] | None = None,
    lowercase: bool = False,
    uppercase: bool = False,
    collapse_whitespace: bool = False,
    unicode_normalize: bool = False,
    verbose: bool = False,
    ignore_missing: bool = True,
) -> pd.DataFrame:
    """
    Clean specified string columns in a dataframe.

    Steps applied to each column:
      * Preserve missing values.
      * Optionally strip characters from both ends (defaults to whitespace).
      * Optionally remove any of the given trim_substrings from start or end.
      * Optionally lowercase.
      * Optionally collapse internal multiple whitespace to single space.
      * (Future) Optionally normalize unicode.

    Args:
        df: pandas DataFrame to clean (not modified in-place; a copy is returned).
        cols: columns to clean; if None or empty, all columns are considered.
        strip_chars: characters to strip from ends (None means default whitespace).
        trim_substrings: substrings to strip from start/end (literal, case-sensitive).
        lowercase: whether to lowercase the result.
        uppercase: whether to uppercae the result.
        collapse_whitespace: collapse internal runs of whitespace to a single space.
        unicode_normalize: if True, apply Unicode normalization (NFC).
        verbose: print before/after for samples.
        ignore_missing: if False, raise if a listed column is missing; if True, skip it.

    Returns
    -------
        A cleaned copy of the dataframe.
    """
    df = df.copy()
    cols_to_process = list(df.columns) if not cols else list(cols)

    # prepare trim substrings, filtering out empties
    trim_list = []
    if trim_substrings:
        trim_list = [s for s in trim_substrings if s]
        if trim_substrings and not trim_list:
            # all were empty; ignore
            trim_list = []

    for col in cols_to_process:
        if col not in df.columns:
            if ignore_missing:
                if verbose:
                    print(f"Warning: column '{col}' not in dataframe; skipping.")
                continue
            raise KeyError(f"Column '{col}' not found in dataframe.")

        if verbose:
            print(f"\n>> Cleaning column '{col}':")
            sample_before = df[col].head(5).tolist()
            print("   sample before:", sample_before)

        # Work on a string-aware series to preserve missingness
        s = df[col].astype("string")  # pandas StringDtype, so <NA> stays as <NA>

        # Strip ends
        s = s.str.strip(strip_chars) if strip_chars is not None else s.str.strip()

        # Trim specified substrings from ends
        if trim_list:
            escaped = [re.escape(x) for x in trim_list]
            pattern = rf"^(?:{'|'.join(escaped)})+|(?:{'|'.join(escaped)})+$"
            s = s.str.replace(pattern, "", regex=True)

        # Lowercase or uppercase if requested
        if lowercase:
            s = s.str.lower()

        if uppercase:
            s = s.str.upper()

        # Collapse internal whitespace
        if collapse_whitespace:
            s = s.str.replace(r"\s+", " ", regex=True)

        # (Optional) Unicode normalization
        if unicode_normalize:
            import unicodedata

            # apply only to non-missing
            s = s.apply(lambda x: unicodedata.normalize("NFC", x) if pd.notna(x) else x)

        df.loc[:, col] = s  # avoid chained assignment warning

        if verbose:
            sample_after = df[col].head(5).tolist()
            print("   sample after: ", sample_after)

    return df


def _cache_paths(source: str) -> dict[str, Path]:
    """Return paths for nodes & edges cache for a given source name."""
    base = CACHE_ROOT / source
    return {
        "nodes": base.with_name(f"{source}_nodes.pkl"),
        "edges": base.with_name(f"{source}_edges.pkl"),
    }


def save_cache(source: str, df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> None:
    """Pickle out nodes & edges DataFrames for this source."""
    paths = _cache_paths(source)
    df_nodes.to_pickle(paths["nodes"])
    df_edges.to_pickle(paths["edges"])


def load_cache(source: str) -> dict[str, pd.DataFrame]:
    """Load pickled nodes & edges; KeyError if missing."""
    paths = _cache_paths(source)
    return {
        "df_nodes": pd.read_pickle(paths["nodes"]),
        "df_edges": pd.read_pickle(paths["edges"]),
    }


def cache_exists(source: str) -> bool:
    """True if both cache files exist for this source."""
    paths = _cache_paths(source)
    return paths["nodes"].exists() and paths["edges"].exists()
