def count_items(item_string, delimiter=',', filter_items=None, return_proportion=False):
    """
    Count occurrences in a delimited string, with optional filtering and proportion calculation.

    Parameters:
    - item_string (str): The string containing items separated by a delimiter.
      If the string is None or empty, it is treated as containing 0 items.
    - delimiter (str): The character used to separate items in the string. Defaults to a comma (',').
    - filter_items (list or set): Optional. A collection of items to filter for.
      Only items in this collection will be counted.
    - return_proportion (bool): If True, returns a tuple of (count, proportion) where proportion is
      the count of matching items over the total item count (e.g., 2/5).

    Returns:
    - int or tuple: If return_proportion is False, returns the count of items in the string
      (after applying filter if provided).
      If return_proportion is True, returns a tuple (count, total, proportion) where total is an integer 
      and proportion is a float.
    """
    # Handle None or empty strings by returning 0 or (0, 0, 0.0) if proportion is requested
    if not item_string:
        return (0, 0, 0.0) if return_proportion else 0

    # Split items by delimiter
    items = item_string.split(delimiter)
    total_items = len(items)

    # Apply filter if specified
    if filter_items:
        matching_items = [item for item in items if item in filter_items]
    else:
        matching_items = items

    count = len(matching_items)

    if return_proportion:
        # Calculate proportion as a float, with total items to avoid division by zero
        proportion = count / total_items if total_items > 0 else 0.0
        return count, total_items, proportion

    return count

    # Example usage
    # example_string = "apple,banana,apple,orange,apple,pineapple,kiwi"
    # Count only 'apple' occurrences, with proportion (note delimiter, so 'apple' won't be counted in pineapple)
    #print(count_items(example_string, filter_items={'apple'}, return_proportion=True))  # Output: (3, 5, 0.6)


def prepare_property_dict_from_df(df, id_col, target_col, operation, property_name, input_list=None):
    """
    Most of the time when we want to provide properties (property_name) to nodes, the nodes will already be in a df (df) of some kind,
    and the properties we would like to assign for each of the nodes (id_col) are in adjacent columns (target_col).
    
    We might not necessarily want to just give the nodes a property however. We might want to give them a property based on that other column before we do.
    For example, while we could use the target_col value as is (use_as_is), we might instead want to test if it is in a list (check_presence_in_list),
    or make sure it is a valid value, i.e. not False or false-like (check_if_not_falsy).

    This function should in many cases help you achieve those checks when making the property dicts, 
    that can be later provided as input to the build_layer() custom_properties argument.

    Note too - if you don't get a filtered_dict, or if it is still very large, you could try first using the check_presence_in_list to get a 
    relevant_ids list based off the vertex_indices for a subgraph. 

    Returns:
    - dict with:
        - property_dict 
        - relevant_ids (only if operation is 'check_presence_in_list' or 'check_if_not_falsy')
        - filtered_df (only if operation is 'check_presence_in_list' or 'check_if_not_falsy')
    """

    helper_dict = {}

    # Create a dictionary of dictionaries in the format of {id_1: {property_name: target_1}, id_2: {property_name: target_2}, etc.}
    if operation=='use_as_is':
        property_dict = dict(df.apply(lambda x: (x[id_col], {property_name: x[target_col]}), axis=1).to_list())
    elif operation=='check_presence_in_list' and len(input_list) > 0:
        property_dict = dict(df.apply(lambda x: (x[id_col], {property_name: True if x[target_col] in input_list else False}), axis=1).to_list())
    elif operation=='check_if_not_falsy':
        property_dict = dict(df.apply(lambda x: (x[id_col], {property_name: False if (not x[target_col]) else True}), axis=1).to_list())
    else:
        raise('Please check operation is supported.')
    helper_dict['property_dict'] = property_dict
    
    # As a helper function for the user, we can filter the df in some cases to only include certain rows of interest
    if operation in ['check_presence_in_list', 'check_if_not_falsy']:
        relevant_ids = [id_ for id_ in property_dict if property_dict[id_][property_name]==True]
        filtered_df = df[df[id_col].isin(relevant_ids)]
    else:
        relevant_ids = None # technically this is *all*
        filtered_df = None # no point returning a filtered df for this, since there may not be anything to filter on - actually, todo: later we could filter out the na values
    helper_dict['relevant_ids'] = relevant_ids
    helper_dict['filtered_df'] = filtered_df

    return helper_dict