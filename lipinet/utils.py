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