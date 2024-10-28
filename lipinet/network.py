from graph_tool.all import Graph, GraphView

class MultilayerNetwork:
    def __init__(self):
        # Initialize the graph
        self.graph = Graph(directed=True)  # Assuming directed edges
        self.node_map = {}  # A dictionary to map node IDs to graph vertices
        
        # Add node properties
        self.layers = self.graph.new_vertex_property("string")  # Layer property (string, like 'layer1', 'layer2')
        self.authority = self.graph.new_vertex_property("string")  # Source of information (ChEBI, Rhea)
        self.node_type = self.graph.new_vertex_property("string")  # Generalized node type (ChEBI, Rhea ID, etc.)
        self.node_id = self.graph.new_vertex_property("string")  # Individual node ID

        # Add properties to the graph
        self.graph.vp['layers'] = self.layers
        self.graph.vp['authority'] = self.authority
        self.graph.vp['node_type'] = self.node_type
        self.graph.vp['node_id'] = self.node_id

        # Add edge properties
        self.edge_weight = self.graph.new_edge_property('float')
        self.edge_layertype = self.graph.new_edge_property('string')

        # Add edge properties to the graph
        self.graph.edge_properties["weight"] = self.edge_weight
        self.graph.edge_properties["interlayer"] = self.edge_layertype


    def add_node(self, layer, authority, node_type, node_id, **extra_properties):
        """
        Add a node to the graph with the given properties.
        """
        # Use (layer, node_id) as the unique key in node_map
        node_key = (layer, node_id)

        # Check if the node already exists; if not, create it
        if node_key not in self.node_map:
            node = self.graph.add_vertex()
            self.node_map[node_key] = node  # Store node with (layer, node_id) as key
        else:
            node = self.node_map[node_key]  # Retrieve existing node

        # Ensure layers is a vertex property map
        if "layers" not in self.graph.vp:
            self.graph.vp["layers"] = self.graph.new_vertex_property("string")
        self.layers = self.graph.vp["layers"]

        # Set core properties
        self.layers[node] = layer
        self.authority[node] = authority
        self.node_type[node] = node_type
        self.node_id[node] = node_id

        # Debugging output to check property setting
        print(f"Node {node_id}: layer={self.layers[node]}, authority={authority}, node_type={node_type}")

        # Set additional properties dynamically
        for prop_name, prop_value in extra_properties.items():
            self._set_node_property(node, prop_name, prop_value)

        return node



    def add_edge_directly(self, node1, node2):
        """
        Add a directed edge between two nodes (keep in mind they should be an actual node object already, else will be created).
        """
        self.graph.add_edge(node1, node2)

    def add_edge_via_nodemap(self, node_pairs):
        """
        Add directed edges between pairs of nodes, using the node map.
        
        Args:
        - node_pairs: A list of tuples (node1_id, node2_id), where each node is identified
                      by (layer, node_id).
        
        Example:
        network.add_edge([(('layer1', 'CHEBI:15377'), ('layer2', 'RHEA:15421'))])
        """
        for node1_id, node2_id in node_pairs:
            node1 = self.node_map.get(node1_id)
            node2 = self.node_map.get(node2_id)
            
            if node1 is None or node2 is None:
                raise ValueError(f"One or both nodes {node1_id}, {node2_id} not found.")
            
            self.graph.add_edge(node1, node2)

    def _set_node_property(self, node, prop_name, value):
        # Check if the property map exists, create it if not
        if prop_name not in self.graph.vp:
            # Determine type based on the value type for simplicity
            prop_type = "string" if isinstance(value, str) else "float" if isinstance(value, float) else "int"
            self.graph.vp[prop_name] = self.graph.new_vertex_property(prop_type)
        
        # Set the property value for the node
        self.graph.vp[prop_name][node] = value

    def build_layer(self, layer_data, layer_name, authority, custom_properties={}):
        """
        Build a network layer from a dataset.
        
        :param layer_data: List of tuples (node_id, node_type)
        :param layer_name: Name of the layer
        :param authority: Source of the data (e.g., ChEBI, Rhea)
        :param custom_properties: Dictionary of extra properties for each node {node_id: {property_name: value}}
        """
        custom_properties = custom_properties or {}
        for node_id, node_type in layer_data:
            self.add_node(layer=layer_name, authority=authority, node_type=node_type, node_id=node_id)
    
    def view_layer(self, layer_name):
        """
        Return a subgraph view of a specific layer.
        """
        return GraphView(self.graph, vfilt=lambda v: self.layers[v] == layer_name)