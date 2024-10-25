import graph_tool.all as gt
from typing import Dict, List


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


# def propagate_categorical_property(
#     g: gt.Graph,
#     measured_node: gt.Vertex,
#     property_prop: gt.PropertyMap,
#     verbose: bool = False,
#     showtotal: bool = True,
# ) -> Dict[str, float]:
#     """
#     Propagates a categorical property from a measured node to its descendants in a 
#     directed graph, accumulating categories at terminal (leaf) nodes, with weights
#     distributed based on the hierarchy structure.

#     Parameters:
#     ----------
#     g : graph_tool.Graph
#         A directed graph object representing the hierarchy between nodes.
#     measured_node : graph_tool.Vertex
#         The node from which the propagation starts. This is the "measured" node of interest.
#     property_prop : graph_tool.PropertyMap
#         A vertex property map (categorical) that assigns properties (categories) to 
#         nodes. The property map is expected to contain lists or arrays of categories for each node.
#     verbose : bool, optional
#         If True, prints the propagation steps for debugging or logging purposes.
#     showtotal : bool, optional
#         If True, prints the total weights of the categories at the terminal nodes.

#     Returns:
#     -------
#     terminal_categories : dict
#         A dictionary where keys are categories (from the property map) and values are 
#         the accumulated weights at terminal nodes (leaf nodes) after the propagation.

#     Example usage:
#     --------------
#     # 1. Create a directed graph and add vertices
#     g = gt.Graph(directed=True)
#     v1 = g.add_vertex()
#     v2 = g.add_vertex()
#     v3 = g.add_vertex()
    
#     # 2. Add edges between vertices (representing a hierarchy)
#     g.add_edge(v1, v2)
#     g.add_edge(v1, v3)
    
#     # 3. Create a vertex property map for categorical properties ('reactions')
#     reactions = g.new_vertex_property("object")
#     reactions[v1] = ['R1']
#     reactions[v2] = ['R2', 'R3']
#     reactions[v3] = []

#     # 4. Propagate categories from a measured node (v1) and accumulate results
#     accumulated_categories = propagate_categorical_property(g, v1, reactions)

#     # 5. Display the accumulated categories and weights
#     print("Accumulated Categories at Terminal Nodes:")
#     for category, weight in accumulated_categories.items():
#         print(f"Category {category}: Accumulated Weight {weight}")

#     # Output Example:
#     # Accumulated Categories at Terminal Nodes:
#     # Category R1: Accumulated Weight 1.0
#     # Category R2: Accumulated Weight 0.5
#     # Category R3: Accumulated Weight 0.5

#     For a more complicated example, the output could look like:
#     # Output Example 2 (full setup not shown):
#     # Accumulated Categories at Terminal Nodes:
#     # Category R2: Accumulated Weight 1.0
#     # Category R3: Accumulated Weight 0.5
#     # Category R4: Accumulated Weight 1.0
#     # Category R5: Accumulated Weight 0.5
#     # Category R9: Accumulated Weight 0.5
#     # Category R10: Accumulated Weight 0.25
#     # Category R6: Accumulated Weight 0.25
#     # Category R7: Accumulated Weight 0.25
#     # Category R8: Accumulated Weight 0.25
#     # Category R11: Accumulated Weight 0.25
#     """
#     if verbose:
#         print(f"Propagating from node {measured_node}")
#     # Initialize a dictionary to store counts of categories at terminal nodes
#     terminal_categories = {} # aka accumulated_categories
    
#     def propagate_categories(node: gt.Vertex, weight: float, inherited_categories: List[str]):
#         """
#         Recursively propagates categories from a node to children, weighting them by the number of children they have.
#         """
#         children = list(node.out_neighbors())
#         num_children = len(children)

#         if verbose:
#             print(f"Processing node {node}, with weight {weight} and inherited categories {inherited_categories}")

#         # If leaf node
#         if num_children == 0:
#             # Combine inherited categories with node's own categories
#             node_categories = inherited_categories.copy()
#             if property_prop[node]:
#                 node_categories.extend(property_prop[node])  # Use extend instead of append
#             # Accumulate category counts with associated weight
#             for category in node_categories:
#                 terminal_categories[category] = terminal_categories.get(category, 0.0) + weight
#             return
#         # Distribute weight among children
#         child_weight = weight / num_children
#         # Inherit categories from parent
#         inherited_categories = inherited_categories.copy()
#         if property_prop[node]:
#             inherited_categories.extend(property_prop[node])  # Use extend instead of append
#         # Propagate to children
#         for child in children:
#             propagate_categories(child, child_weight, inherited_categories)
    
#     # Start propagation from measured node with initial weight of 1.0
#     propagate_categories(measured_node, 1.0, [])

#     # Display the results
#     if showtotal:
#         print(f"Accumulated Categories at Terminal Nodes (from node {measured_node}):")
#         for category, weight in terminal_categories.items():
#             print(f"Category {category}: Accumulated Weight {weight}")
    
#     # Return the accumulated categories
#     return terminal_categories