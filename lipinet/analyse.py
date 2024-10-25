import graph_tool.all as gt
from typing import Dict, List

# import helper functions
from lipinet.utils import count_items


def propagate_categorical_property(
    g: gt.Graph,
    measured_node: gt.Vertex,
    property_prop: gt.PropertyMap,
    direction: str = 'downstream',
    verbose: bool = False,
    showtotal: bool = True,
    storepath: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Propagates a categorical property from a measured node, tracking the origin
    of each category (upstream, measured node, downstream) and accumulating weights.

    Returns:
    -------
    terminal_categories : dict
        A dictionary with keys as sources ('upstream', 'measured_node', 'downstream') and
        values as dictionaries mapping categories to their accumulated weights.
    """
    if verbose:
        print(f"Propagating from node {measured_node} in direction '{direction}'")
    # Initialize dictionaries to store categories based on their origin
    terminal_categories = {
        'upstream': {},
        'measured_node': {},
        'downstream': {}
    }

    # Keep track of visited nodes to prevent infinite loops
    visited = set()

    # Keep the path / log if requested
    pathlog = []
    def store_this_path(update_node, update_weight, update_origin, update_inherited_categories, update_current_category):
        # Keep the path / log if requested
        pathlog.append({'node':int(update_node), 'weight':update_weight, 
                        'current_origin':update_origin,
                        'inherited_categories':update_inherited_categories,
                        'current_category':update_current_category})


    def propagate_categories(node: gt.Vertex, weight: float, inherited_categories: Dict[str, str], current_origin: str):
        if node in visited:
            return
        visited.add(node)

        if verbose:
            print(f"Processing node {int(node)}, with weight {weight} and inherited categories {inherited_categories}")

        # Copy the inherited categories to avoid mutating the parent's data
        current_inherited_categories = inherited_categories.copy()

        # Add categories from the current node
        node_categories = property_prop[node]
        if node_categories:
            for category in node_categories:
                if category not in current_inherited_categories: 
                    if current_origin == 'start': # Assign origin if category is new or keep existing origin - note: this led to a double weighted measured node...
                    #if int(node) == int(measured_node): # Assign origin based on where the category is first encountered
                        current_inherited_categories[category] = 'measured_node'
                    else:
                        current_inherited_categories[category] = current_origin
                    # Potentially store the path, if the category has just been updated - note this will represent in two or more log entries where there are nested properties for the one node (e.g. multiple reactions ["R1","R2"])
                    if storepath: store_this_path(node, weight, current_origin, inherited_categories, category)
                else:
                    # Potentially store the path, even if the category hasn't just been updated 
                    if storepath: store_this_path(node, weight, current_origin, inherited_categories, None)

        # Determine neighbors based on the direction
        if direction == 'downstream':
            neighbors = list(node.out_neighbors())
            num_neighbors = len(neighbors)
            # If terminal node (no downstream neighbors)
            if num_neighbors == 0:
                # Accumulate categories at terminal node
                for category, origin in current_inherited_categories.items():
                    terminal_categories[origin][category] = terminal_categories[origin].get(category, 0.0) + weight
                return
            # Distribute weight among children
            child_weight = weight / num_neighbors if num_neighbors > 0 else weight
            # Propagate to children
            for neighbor in neighbors:
                propagate_categories(neighbor, child_weight, current_inherited_categories, 'downstream')

        elif direction == 'upstream':
            neighbors = list(node.in_neighbors())
            num_neighbors = len(neighbors)
            # If terminal node (no upstream neighbors)
            if num_neighbors == 0:
                # Accumulate categories at terminal node
                for category, origin in current_inherited_categories.items():
                    terminal_categories[origin][category] = terminal_categories[origin].get(category, 0.0) + weight
                return
            # Do not divide weight among parents
            for neighbor in neighbors:
                propagate_categories(neighbor, weight, current_inherited_categories, 'upstream')

        elif direction == 'both':
            downstream_neighbors = list(node.out_neighbors())
            upstream_neighbors = list(node.in_neighbors())
            num_downstream = len(downstream_neighbors)
            num_upstream = len(upstream_neighbors)

            is_leaf = num_downstream == 0
            is_root = num_upstream == 0

            # Accumulate categories at terminal nodes
            if is_leaf or is_root:
                for category, origin in current_inherited_categories.items():
                    terminal_categories[origin][category] = terminal_categories[origin].get(category, 0.0) + weight

            # Propagate downstream (divide weight among children)
            if num_downstream > 0:
                child_weight = weight / num_downstream
                for neighbor in downstream_neighbors:
                    propagate_categories(neighbor, child_weight, current_inherited_categories, 'downstream')

            # Propagate upstream (do not divide weight)
            if num_upstream > 0:
                for neighbor in upstream_neighbors:
                    propagate_categories(neighbor, weight, current_inherited_categories, 'upstream')

        else:
            raise ValueError("Invalid direction. Choose 'downstream', 'upstream', or 'both'.")

    # Start propagation from measured node with initial weight of 1.0
    # The origin at the start node is 'measured_node'
    propagate_categories(measured_node, 1.0, {}, 'start')

    # Manual correction of the measured node to stop the weights being duplicated (i.e. set to 2.0)...
    terminal_categories['measured_node'] = {category:1.0 for category in terminal_categories['measured_node']}

    if storepath:
        terminal_categories['pathlog'] = pathlog

    # Display the results
    if showtotal:
        print(f"Accumulated Categories at Terminal Nodes (from node {measured_node}):")
        for origin, categories in terminal_categories.items():
            if origin=='pathlog':
                continue #skip printing the pathlog if it has been included
            print(f"\nCategories from {origin.replace('_', ' ')}:")
            for category, weight in categories.items():
                print(f"  Category {category}: Accumulated Weight {weight}")

    # Return the accumulated categories
    return terminal_categories


# Example usage in graph property assignment
def add_count_property(graph, property_name, new_property_name, delimiter=','):
    """
    Adds a property to a graph that counts items in a specified property with a delimiter.

    Parameters:
    - graph (Graph): The graph to which the property will be added.
    - property_name (str): The name of the property containing the delimited strings.
    - new_property_name (str): The name of the new property to store the counts.
    - delimiter (str): The delimiter used to separate items in the property strings.

    Returns:
    - PropertyMap: The new property map with counts of items for each vertex.
    """
    # Create a new vertex property of type double for counting
    item_count_property = graph.new_vertex_property("double")
    
    for v in graph.vertices():
        item_count_property[v] = count_items(graph.vp[property_name][v], delimiter=delimiter)
    
    # Assign the new property to the graph
    graph.vp[new_property_name] = item_count_property
    return item_count_property

    # Example of how to use the functions
    # Assume `g` is an existing graph with a vertex property 'items' that contains delimited strings
    # v_item_count = add_count_property(g, 'reactions', 'item_count')