from graph_tool.all import Graph, GraphView, bfs_search, BFSVisitor, bfs_iterator, shortest_distance, graph_draw
from graph_tool.topology import label_out_component
from collections import deque


class MultilayerNetwork:
    def __init__(self):
        # Initialize the graph
        self.graph = Graph(directed=True)  # Assuming directed edges
        self.node_map = {}  # A dictionary to map node IDs to graph vertices
        
        # Add node properties
        self.layer = self.graph.new_vertex_property("string")  # Layer property (string, like 'layer1', 'layer2')
        self.authority = self.graph.new_vertex_property("string")  # Source of information (ChEBI, Rhea)
        self.node_type = self.graph.new_vertex_property("string")  # Generalized node type (ChEBI, Rhea ID, etc.)
        self.node_id = self.graph.new_vertex_property("string")  # Individual node ID

        # Add properties to the graph
        self.graph.vp['layer'] = self.layer
        self.graph.vp['authority'] = self.authority
        self.graph.vp['node_type'] = self.node_type
        self.graph.vp['node_id'] = self.node_id

        # Add edge properties
        self.edge_weight = self.graph.new_edge_property('float')
        self.edge_layertype = self.graph.new_edge_property('string')

        # Add edge properties to the graph
        self.graph.edge_properties["weight"] = self.edge_weight
        self.graph.edge_properties["interlayer"] = self.edge_layertype


    def add_node(self, layer, authority, node_type, node_id, verbose=True, **extra_properties):
        """
        Add a node to the graph with the given properties and additional custom properties.
        
        :param layer: Layer name for the node.
        :param authority: Source authority for the node.
        :param node_type: Type of the node.
        :param node_id: Unique identifier for the node.
        :param extra_properties: Additional properties for the node.
        """
        # Use (layer, node_id) as the unique key in node_map
        node_key = (layer, node_id)

        # Check if the node already exists; if not, create it
        if node_key not in self.node_map:
            node = self.graph.add_vertex()
            self.node_map[node_key] = node  # Store node with (layer, node_id) as key
        else:
            node = self.node_map[node_key]  # Retrieve existing node

        # Set core properties using _set_node_property
        self._set_node_property(node, "layer", layer) # previously adjusted like this: self.layer[node] = layer
        self._set_node_property(node, "authority", authority)
        self._set_node_property(node, "node_type", node_type)
        self._set_node_property(node, "node_id", node_id)

        if verbose:
            # Debugging output to check property setting
            print(f"Node {node_id}: layer={layer}, authority={authority}, node_type={node_type}, extra props={extra_properties.items()}")

        # Set additional custom properties
        for prop_name, prop_value in extra_properties.items():
            self._set_node_property(node, prop_name, prop_value)



    def add_edge_directly(self, node1, node2):
        """
        Add a directed edge between two nodes (keep in mind they should be an actual node object already, else will be created).
        """
        self.graph.add_edge(node1, node2)


    def add_edge_via_nodemap(self, node_pairs=None, from_nodes=None, to_nodes=None, 
                            from_layer=None, to_layer=None, split_char='|', create_missing=False):
        """
        Add directed edges between pairs of nodes, using the node map.

        Args:
        - node_pairs: A list of tuples (node1_id, node2_id), where each node is identified
                    by (layer, node_id). node_id can include multiple IDs separated by `split_char`.
        - from_nodes: A list, series, or array of node IDs for the 'from' nodes.
        - to_nodes: A list, series, or array of node IDs for the 'to' nodes.
        - from_layer: Layer name to use for all nodes in from_nodes (required if using from_nodes/to_nodes).
        - to_layer: Layer name to use for all nodes in to_nodes (required if using from_nodes/to_nodes).
        - split_char: Optional; the character used to split multiple node IDs (default is '|').
        - create_missing: Optional; if True, creates missing nodes if they are not found in node_map.

        Examples:

        Using node_pairs:
            network.add_edge_via_nodemap(
                node_pairs=[
                    (('layer1', 'CHEBI:15377'), ('layer2', 'RHEA:15421 | CHEBI:15378')),
                    (('layer1', 'CHEBI:15379'), ('layer2', 'CHEBI:15380 | CHEBI:15381')),
                    (('layer1', 'CHEBI:15382'), ('layer2', 'RHEA:15422'))
                ],
                split_char='|',
                create_missing=True
            )

        Using from_nodes and to_nodes:
            network.add_edge_via_nodemap(
                from_nodes=['CHEBI:15377', 'CHEBI:15378'],
                to_nodes=['RHEA:15421 | CHEBI:15379', 'RHEA:15420'],
                from_layer='layer1',
                to_layer='layer2',
                split_char='|',
                create_missing=True
            )
        """
        def get_or_create_node(layer, node_id):
            """
            Helper function to get a node from node_map or create it if missing.
            """
            node_key = (layer, node_id)
            node = self.node_map.get(node_key)
            
            if node is None and create_missing:
                print(f"Node {node_key} not found. Creating node.")
                node = self.graph.add_vertex()
                self.node_map[node_key] = node
                # Set default properties if needed (e.g., layer and node_id)
                self.layer[node] = layer
                self.node_id[node] = node_id
                
            return node

        if node_pairs is not None:
            # Process node_pairs as before
            for node1_id, node2_id in node_pairs:
                if node1_id is None or node2_id is None:
                    continue  # Skip null entries
                
                # Check if node IDs contain the split character, and split & strip spaces if they do
                node1_ids = [id_.strip() for id_ in node1_id[1].split(split_char)] if isinstance(node1_id[1], str) and split_char in node1_id[1] else [node1_id[1]]
                node2_ids = [id_.strip() for id_ in node2_id[1].split(split_char)] if isinstance(node2_id[1], str) and split_char in node2_id[1] else [node2_id[1]]

                # Iterate through all combinations of node1_ids and node2_ids
                for id1 in node1_ids:
                    for id2 in node2_ids:
                        node1 = get_or_create_node(node1_id[0], id1)
                        node2 = get_or_create_node(node2_id[0], id2)

                        # Skip if nodes couldn't be created or retrieved
                        if node1 is None or node2 is None:
                            continue
                        self.graph.add_edge(node1, node2)
        elif from_nodes is not None and to_nodes is not None:
            # Ensure from_layer and to_layer are provided
            if from_layer is None or to_layer is None:
                raise ValueError("Both from_layer and to_layer must be specified when using from_nodes and to_nodes.")

            # Process from_nodes and to_nodes with specified layers
            for from_node_id, to_node_id in zip(from_nodes, to_nodes):
                # Skip null entries and non-string types
                if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
                    print(f"Skipping null or non-string entry: from_node_id={from_node_id}, to_node_id={to_node_id}")
                    continue
                
                from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
                to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

                for id1 in from_node_ids:
                    for id2 in to_node_ids:
                        node1 = get_or_create_node(from_layer, id1)
                        node2 = get_or_create_node(to_layer, id2)

                        # Skip if nodes couldn't be created or retrieved
                        if node1 is None or node2 is None:
                            continue
                        self.graph.add_edge(node1, node2)
        else:
            raise ValueError("Either node_pairs or both from_nodes and to_nodes must be provided.")




    def _set_node_property(self, node, prop_name, value, nan_replacement=None):
        """
        Set or create a node property for a given node.
        
        :param node: The node (vertex) to assign the property.
        :param prop_name: The name of the property.
        :param value: The value of the property, which determines the property type.
        :param nan_replacement: Optional; the value to replace NaN with (default is None).
        """
        import math
        # Replace NaN with the specified replacement, if value is NaN
        if isinstance(value, float) and math.isnan(value):
            value = nan_replacement

        # Check if the property map exists, create it if not
        if prop_name not in self.graph.vp:
            # Determine type based on the value type for simplicity
            if isinstance(value, str):
                prop_type = "string"
            elif isinstance(value, float):
                prop_type = "float"
            elif isinstance(value, int):
                prop_type = "int"
            else:
                prop_type = "object"  # For more complex data types

            self.graph.vp[prop_name] = self.graph.new_vertex_property(prop_type)
        
        # Set the property value for the node
        self.graph.vp[prop_name][node] = value


    def build_layer(self, layer_data, layer_name, authority, custom_properties={}, verbose=True):
        """
        Build a network layer from a dataset.
        
        :param layer_data: List of tuples (node_id, node_type)
        :param layer_name: Name of the layer
        :param authority: Source of the data (e.g., ChEBI, Rhea)
        :param custom_properties: Dictionary of extra properties for each node {node_id: {property_name: value}}
        :param verbose: Whether to print every node update.
        """
        custom_properties = custom_properties or {}
        for node_id, node_type in layer_data:
            # Get extra properties for this node if they exist
            node_properties = custom_properties.get(node_id, {})
            self.add_node(layer=layer_name, authority=authority, node_type=node_type, node_id=node_id, verbose=verbose, **node_properties)
    
    
    def view_layer(self, layer_name):
        """
        Return a subgraph view of a specific layer.
        """
        return GraphView(self.graph, vfilt=lambda v: self.layer[v] == layer_name)
    

    def filter_view_by_property(self, prop_name, target_value, comparison="=="):
        """
        Creates a filtered graph view including only nodes where the specified property
        meets the specified comparison with the target value.

        Note, the other way to do something similar to graph tool is using the graph_tool.util.find_vertex()
        In some cases you may want to use that instead. e.g:
            graph_tool.util.find_vertex(g=multinet.graph, prop=multinet.graph.vertex_properties['layer'], match='swisslipids')

        :param prop_name: The name of the property to filter by.
        :param target_value: The value to filter for.
        :param comparison: The comparison operator as a string (e.g., "==", "!=", "<", ">").
        :return: A GraphView filtered to include only nodes where `prop_name comparison target_value`.
        """
        import operator

        # Define available operators
        comparison_operators = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge
        }
        
        # Check if the property exists in vertex properties
        if prop_name not in self.graph.vp:
            raise ValueError(f"Property '{prop_name}' does not exist in the graph.")

        # Check if the comparison operator is valid
        if comparison not in comparison_operators:
            raise ValueError(f"Invalid comparison operator '{comparison}'. Choose from {list(comparison_operators.keys())}.")

        # Get the appropriate comparison function
        compare_func = comparison_operators[comparison]

        # Create a filter mask based on the property and comparison
        def filter_func(v):
            return compare_func(self.graph.vp[prop_name][v], target_value)

        # Create and return a filtered GraphView
        return GraphView(self.graph, vfilt=filter_func)
    

    def extract_subgraph_with_paths(self, root, nodes_of_interest, direction='upstream'):
        """
        Extracts a subgraph that includes the nodes of interest, the root,
        and any intermediate nodes along the paths from each node of interest according to the specified direction.
        Direction can be 'upstream', 'downstream', or 'both'.

        :param root: The root node from which paths are calculated.
        :param nodes_of_interest: A set of nodes of interest to include in the subgraph.
        :param direction: The direction of traversal: 'upstream', 'downstream', or 'both'.
        :return: A GraphView of the subgraph containing relevant nodes and edges.
        """
        # Ensure `root` is a Vertex object
        if isinstance(root, int):
            root = self.graph.vertex(root)

        # Ensure `nodes_of_interest` are Vertex objects
        nodes_of_interest_vertices = {self.graph.vertex(idx) for idx in nodes_of_interest}

        # Initialize global vertex and edge filters
        vfilt = self.graph.new_vertex_property('bool', val=False)
        efilt = self.graph.new_edge_property('bool', val=False)

        if direction in ['upstream', 'both']:
            # Upstream traversal (towards ancestors)
            self._bfs_traversal(nodes_of_interest_vertices, vfilt, efilt, mode='upstream')
            # Include the root node in upstream traversal
            vfilt[root] = True

        if direction in ['downstream', 'both']:
            # Downstream traversal (towards descendants)
            self._bfs_traversal(nodes_of_interest_vertices, vfilt, efilt, mode='downstream')

        # Create a GraphView with the combined filters
        subgraph = GraphView(self.graph, vfilt=vfilt, efilt=efilt)
        return subgraph

    def _bfs_traversal(self, seed_vertices, vfilt, efilt, mode='downstream'):
        """
        Performs BFS traversal from seed vertices in the specified mode ('upstream' or 'downstream').
        Updates the provided vertex and edge filters.

        :param seed_vertices: Set of seed Vertex objects.
        :param vfilt: Vertex property map to update.
        :param efilt: Edge property map to update.
        :param mode: 'upstream' for ancestors, 'downstream' for descendants.
        """
        visited = set()
        queue = deque(seed_vertices)

        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            vfilt[v] = True

            if mode == 'downstream':
                # Traverse outgoing edges
                for e in v.out_edges():
                    target = e.target()
                    efilt[e] = True
                    if target not in visited:
                        queue.append(target)
            elif mode == 'upstream':
                # Traverse incoming edges
                for e in v.in_edges():
                    source = e.source()
                    efilt[e] = True
                    if source not in visited:
                        queue.append(source)
            else:
                raise ValueError("Mode must be 'upstream' or 'downstream'.")
    

    

    def search(self, start_node_idx, max_dist=5, direction='downstream', node_text='ids', show_plot=True, **kwargs):
        """
        Generalized function to perform upstream, downstream, or both-directional search on a directed graph.
        
        Parameters:
        - start_node_idx: The index of the node to start the search from.
        - max_dist: Maximum distance (in number of hops) to search.
        - direction: 'downstream', 'upstream', or 'both' to search in both directions.
        - node_text: Attribute to display on nodes ('ids' or 'node_id').
        - show_plot: Boolean to show plot or not.
        
        Returns:
        - A filtered subgraph containing the nodes within the given distance in the specified direction.
        """
        g = self.graph
        MAX_DIST = max_dist

        # Step 1: Select the starting node
        start_node = g.vertex(start_node_idx)

        # Step 2: Handle direction (upstream, downstream, or both)
        if direction == 'upstream':
            # Reverse the graph for upstream search
            g_search = g.copy()
            g_search.set_reversed(True)
            distances = shortest_distance(g_search, source=start_node, max_dist=MAX_DIST)

        elif direction == 'downstream':
            # Use the graph as is for downstream search
            distances = shortest_distance(g, source=start_node, max_dist=MAX_DIST)

        elif direction == 'both':
            # Perform both upstream and downstream searches separately
            # Upstream search with reversed graph
            g_upstream = g.copy()
            g_upstream.set_reversed(True)
            distances_upstream = shortest_distance(g_upstream, source=start_node, max_dist=MAX_DIST)
            
            # Downstream search
            distances_downstream = shortest_distance(g, source=start_node, max_dist=MAX_DIST)
            
            # Merge distances (take minimum distance if reachable in both directions)
            distances = {v: min(distances_upstream[v], distances_downstream[v]) 
                        for v in g.vertices() 
                        if distances_upstream[v] < float('inf') or distances_downstream[v] < float('inf')}

        else:
            raise ValueError("Invalid direction. Choose 'upstream', 'downstream', or 'both'.")

        # Step 3: Filter the graph to only include nodes within the specified distance
        result_filter = GraphView(g, vfilt=lambda v: distances[v] <= MAX_DIST and distances[v] < float('inf'))

        # Output details
        print(f"{direction.capitalize()} graph from node {start_node} contains {result_filter.num_vertices()} vertices and {result_filter.num_edges()} edges.")

        # Optionally draw the filtered graph
        if show_plot:
            vertex_text_prop = result_filter.vertex_properties['node_id'] if node_text == 'node_id' else result_filter.vertex_index
            graph_draw(result_filter, vertex_text=vertex_text_prop, **kwargs)

        return result_filter
    

    def extract_subgraph_with_label_component(self, root, nodes_of_interest, direction='downstream'):
        """
        Extracts a subgraph including all nodes reachable from nodes of interest 
        in the specified direction ('upstream', 'downstream', or 'both').

        Parameters:
        - root: The root node from which paths are calculated.
        - nodes_of_interest: A set of node indices of interest to include in the subgraph.
        - direction: The direction of traversal ('upstream', 'downstream', or 'both').

        Returns:
        - A GraphView of the subgraph containing relevant nodes and edges.
        """
        # Initialize an empty vertex filter property map with False values
        vfilt = self.graph.new_vertex_property('bool', val=False)
        
        # Downstream component
        if direction in ['downstream', 'both']:
            for node_idx in nodes_of_interest:
                node = self.graph.vertex(node_idx)
                component = label_out_component(self.graph, node)
                vfilt.a = vfilt.a | component.a  # Logical OR to combine component labels

        # Upstream component by using a reversed graph view
        if direction in ['upstream', 'both']:
            reversed_graph = GraphView(self.graph, reversed=True)
            for node_idx in nodes_of_interest:
                node = reversed_graph.vertex(node_idx)
                component = label_out_component(reversed_graph, node)
                vfilt.a = vfilt.a | component.a  # Logical OR to combine component labels

        # Include the root node explicitly
        vfilt[root] = True

        # Create the filtered subgraph with the combined vertex filter
        subgraph = GraphView(self.graph, vfilt=vfilt)
        return subgraph
        

    def inspect_properties(self, vertex, verbose=True):
        node_properties = {prop_name: prop[vertex] for prop_name, prop in self.graph.vertex_properties.items()}
        if verbose:
            print(node_properties)
        return node_properties


def get_root_nodes(graph):
    """
    Speculative root nodes. Should refactor later to allow for different conditions or graph conventions.
    """
    root_nodes = [v for v in graph.vertices() if v.in_degree() == 0]
    return root_nodes

