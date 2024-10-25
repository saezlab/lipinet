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

    def add_node(self, layer, authority, node_type, node_id):
        """
        Add a node to the graph with the given properties.
        """
        v = self.graph.add_vertex()  # Add vertex
        self.node_map[(layer, node_id)] = v # Add to the node map
        self.layers[v] = layer
        self.authority[v] = authority
        self.node_type[v] = node_type
        self.node_id[v] = node_id
        return v

    def add_edge_directly(self, node1, node2):
        """
        Add a directed edge between two nodes.
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

    def build_layer(self, layer_data, layer_name, authority):
        """
        Build a network layer from a dataset.
        
        :param layer_data: List of tuples (node_id, node_type)
        :param layer_name: Name of the layer
        :param authority: Source of the data (e.g., ChEBI, Rhea)
        """
        for node_id, node_type in layer_data:
            self.add_node(layer=layer_name, authority=authority, node_type=node_type, node_id=node_id)
    
    def view_layer(self, layer_name):
        """
        Return a subgraph view of a specific layer.
        """
        return GraphView(self.graph, vfilt=lambda v: self.layers[v] == layer_name)