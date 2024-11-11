from graph_tool.all import Graph, GraphView, bfs_search, BFSVisitor, bfs_iterator, shortest_distance, graph_draw
from graph_tool.topology import label_out_component
from collections import deque

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from graph_tool.all import Graph, graph_draw
import json
import math

from itertools import product


class MultilayerNetwork:
    def __init__(self):
        # Initialize the graph
        # self.graph = Graph(directed=True)  # Assuming directed edges
        # self.node_map = {}  # A dictionary to map node IDs to graph vertices
        
        # # Add node properties
        # self.layer = self.graph.new_vertex_property("string")  # Layer property (string, like 'layer1', 'layer2')
        # self.authority = self.graph.new_vertex_property("string")  # Source of information (ChEBI, Rhea)
        # self.node_type = self.graph.new_vertex_property("string")  # Generalized node type (ChEBI, Rhea ID, etc.)
        # self.node_id = self.graph.new_vertex_property("string")  # Individual node ID

        # # Add properties to the graph
        # self.graph.vp['layer'] = self.layer
        # self.graph.vp['authority'] = self.authority
        # self.graph.vp['node_type'] = self.node_type
        # self.graph.vp['node_id'] = self.node_id

        # # Add edge properties
        # self.edge_weight = self.graph.new_edge_property('float')
        # self.edge_layertype = self.graph.new_edge_property('string')

        # # Add edge properties to the graph
        # self.graph.edge_properties["weight"] = self.edge_weight
        # self.graph.edge_properties["edge_layertype"] = self.edge_layertype

        ### REFACTOR BELOW

        self.graph = Graph(directed=True)
        self.node_map = {}  # Maps (layer, node_id) to vertex

        # Define vertex properties
        self.vp_node_id = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["node_id"] = self.vp_node_id

        self.vp_layer = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["layer"] = self.vp_layer

        # Define additional node properties (Prop1 to Prop10)
        # for i in range(1, 11):
        #     prop = f'Prop{i}'
        #     self.graph.vertex_properties[prop] = self.graph.new_vertex_property("string")

        # # Define edge properties (EdgeProp1 to EdgeProp10)
        # for i in range(1, 11):
        #     prop = f'EdgeProp{i}'
        #     self.graph.edge_properties[prop] = self.graph.new_edge_property("float")

        # Edge properties will be added dynamically as needed

    def _ensure_vertex_property(self, prop_name, prop_type="string"):
        """
        Ensure that a vertex property exists. If not, create it.

        :param prop_name: Name of the property.
        :param prop_type: Type of the property (default: 'string').
        """
        if prop_name not in self.graph.vertex_properties:
            self.graph.vertex_properties[prop_name] = self.graph.new_vertex_property(prop_type)

    def _ensure_edge_property(self, prop_name, prop_type="float"):
        """
        Ensure that an edge property exists. If not, create it.

        :param prop_name: Name of the property.
        :param prop_type: Type of the property (default: 'float').
        """
        if prop_name not in self.graph.edge_properties:
            self.graph.edge_properties[prop_name] = self.graph.new_edge_property(prop_type)

    def _add_node(self, layer, node_id, properties=None):
        """
        Add a node with the given layer and node_id, and assign properties.

        :param layer: The layer of the node.
        :param node_id: The identifier of the node.
        :param properties: A dictionary of property_name: value pairs.
        :return: The vertex object.
        """
        key = (layer, node_id)
        if key in self.node_map:
            return self.node_map[key]
        
        v = self.graph.add_vertex()
        self.vp_node_id[v] = node_id
        self.vp_layer[v] = layer if layer else 'Unknown'

        if properties:
            for prop, value in properties.items():
                # Determine property type based on value
                if isinstance(value, float):
                    prop_type = "float"
                elif isinstance(value, int):
                    prop_type = "int"
                else:
                    prop_type = "string"

                self._ensure_vertex_property(prop, prop_type)
                self.graph.vertex_properties[prop][v] = value if not (isinstance(value, float) and math.isnan(value)) else "Unknown"

        self.node_map[key] = v
        return v

    def add_edges_from_dataframe(
        self,
        df: pd.DataFrame,
        from_node_id_col: str,
        from_layer_col: str,
        to_node_id_col: str,
        to_layer_col: str,
        edge_property_cols: list = None,
        from_node_property_cols: list = None,
        to_node_property_cols: list = None,
        split_char: str = '|',
        create_missing: bool = False,
        skip_if_duplicate: str = 'exact',
        verbose: bool = True,
        edge_property_mode: str = 'per-edge',  # 'per-edge' or 'per-row'
        null_node_id: str = 'skip'  # Options: 'skip', 'set_to_missing', or custom string
    ):
        """
        Adds directed edges and nodes from a single DataFrame, assigning properties appropriately.
        Automatically handles node creation and null node IDs.

        Args:
        - df (pd.DataFrame): DataFrame containing edge and node information.
        - from_node_id_col (str): Column name in df for source node IDs.
        - from_layer_col (str): Column name in df for source node layers.
        - to_node_id_col (str): Column name in df for target node IDs.
        - to_layer_col (str): Column name in df for target node layers.
        - edge_property_cols (list of str, optional): List of column names in df to be used as edge properties.
        - from_node_property_cols (list of str, optional): List of column names in df to be used as properties for source nodes.
        - to_node_property_cols (list of str, optional): List of column names in df to be used as properties for target nodes.
        - split_char (str, optional): Character used to split multiple node IDs within a single cell. Default is '|'.
        - create_missing (bool, optional): If True, creates missing nodes when they are not found in node_map. Default is False.
        - skip_if_duplicate (str, optional): Strategy to handle duplicate edges ('exact', 'any', or None). Default is 'exact'.
        - verbose (bool, optional): If True, prints debug information. Default is True.
        - edge_property_mode (str, optional): Mode for handling edge properties.
            - 'per-edge': Each edge has its own property value. Requires property lists to match the number of edges.
            - 'per-row': Each row provides a single property value applied to all edges generated from that row.
            Default is 'per-edge'.
        - null_node_id (str, optional): Strategy for handling null or blank node IDs.
            - 'skip': Exclude nodes with null or blank 'node_id'.
            - 'set_to_missing': Assign a default label like 'Missing'.
            - Custom string: Assign a user-defined label (e.g., 'Unknown').
            Default is 'skip'.

        Raises:
        - ValueError: If edge_property_mode is invalid or required layer columns are missing.
        """
        # Validate edge_property_mode
        if edge_property_mode not in ['per-edge', 'per-row']:
            raise ValueError("`edge_property_mode` must be either 'per-edge' or 'per-row'.")

        # Validate input DataFrame
        required_cols = [from_node_id_col, from_layer_col, to_node_id_col, to_layer_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame.")

        # Determine edge properties to use
        excluded_cols = {from_node_id_col, from_layer_col, to_node_id_col, to_layer_col}
        if edge_property_cols:
            excluded_cols.update(edge_property_cols)
        if from_node_property_cols:
            excluded_cols.update(from_node_property_cols)
        if to_node_property_cols:
            excluded_cols.update(to_node_property_cols)

        # If edge_property_cols is not specified, exclude node-related columns
        if edge_property_cols is None:
            property_cols = [col for col in df.columns if col not in excluded_cols]
        else:
            property_cols = edge_property_cols

        # Prepare edge properties by extracting specified columns as lists
        edge_properties = {}
        for col in property_cols:
            if col in df.columns:
                edge_properties[col] = df[col].tolist()
            else:
                raise ValueError(f"Edge property column '{col}' not found in DataFrame.")

        # Handle edge_property_mode
        if edge_property_mode == 'per-edge':
            if verbose:
                print("Calculating the total number of edges for 'per-edge' mode...")
            df['from_nodes_split'] = df[from_node_id_col].apply(
                lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
            )
            df['to_nodes_split'] = df[to_node_id_col].apply(
                lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
            )
            df['num_edges'] = df.apply(lambda row: len(row['from_nodes_split']) * len(row['to_nodes_split']), axis=1)
            expected_num_edges = df['num_edges'].sum()

            # Validate that each edge property list matches the total number of edges
            self.validate_edge_properties_length(edge_properties, expected_num_edges)

        elif edge_property_mode == 'per-row':
            if verbose:
                print("Validating edge properties for 'per-row' mode...")
            num_valid_rows = len(df.dropna(subset=[from_node_id_col, to_node_id_col]))
            for prop_name, prop_value in edge_properties.items():
                if isinstance(prop_value, list):
                    if len(prop_value) != len(df):
                        raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match the number of rows ({len(df)}).")
                else:
                    # If single value, replicate it for all rows
                    edge_properties[prop_name] = [prop_value] * len(df)

    #     # [Previous validation code remains the same]

    #     if verbose:
    #         print("Processing DataFrame to generate edge combinations...")

    #     # Assign new columns without copying
    #     df['from_nodes_split'] = df[from_node_id_col].str.split(split_char)
    #     df['to_nodes_split'] = df[to_node_id_col].str.split(split_char)

    #     # Explode both columns
    #     df_exploded = df.explode('from_nodes_split')
    #     df_exploded = df_exploded.explode('to_nodes_split').reset_index(drop=True)

    #     # Handle null or blank node IDs
    #     df_exploded['from_nodes'] = df_exploded['from_nodes_split'].replace('', np.nan)
    #     df_exploded['to_nodes'] = df_exploded['to_nodes_split'].replace('', np.nan)

    #     if null_node_id == 'set_to_missing':
    #         df_exploded['from_nodes'] = df_exploded['from_nodes'].fillna('Missing')
    #         df_exploded['to_nodes'] = df_exploded['to_nodes'].fillna('Missing')
    #     elif null_node_id == 'skip':
    #         df_exploded.dropna(subset=['from_nodes', 'to_nodes'], inplace=True)

    #     # Create unique node DataFrame
    #     unique_from = df_exploded[[from_layer_col, 'from_nodes']].drop_duplicates()
    #     unique_to = df_exploded[[to_layer_col, 'to_nodes']].drop_duplicates()
    #     unique_nodes = pd.concat([
    #         unique_from.rename(columns={from_layer_col: 'layer', 'from_nodes': 'node_id'}),
    #         unique_to.rename(columns={to_layer_col: 'layer', 'to_nodes': 'node_id'})
    #     ]).drop_duplicates()

    #     # Assign vertex indices
    #     unique_nodes['vertex_index'] = range(self.graph.num_vertices(), self.graph.num_vertices() + len(unique_nodes))

    #     # Bulk-add vertices
    #     self.graph.add_vertex(len(unique_nodes))

    #     # Update node_map
    #     self.node_map.update({
    #         (row['layer'], row['node_id']): int(row['vertex_index'])
    #         for _, row in unique_nodes.iterrows()
    #     })

    #     # Assign properties in bulk
    #     self._assign_vertex_properties(
    #         unique_nodes, 
    #         from_node_property_cols, 
    #         to_node_property_cols, 
    #         df_exploded, 
    #         from_layer_col, 
    #         to_layer_col
    #     )

    #     # Map node keys to vertex indices
    #     df_exploded['from_key'] = list(zip(df_exploded[from_layer_col], df_exploded['from_nodes']))
    #     df_exploded['to_key'] = list(zip(df_exploded[to_layer_col], df_exploded['to_nodes']))

    #     from_vertices = df_exploded['from_key'].map(self.node_map).astype(int)
    #     to_vertices = df_exploded['to_key'].map(self.node_map).astype(int)

    #     # Prepare edge list
    #     edge_list = np.column_stack((from_vertices, to_vertices))

    #     # Add edges
    #     self.graph.add_edge_list(edge_list)

    #     # Assign edge properties
    #     if edge_property_cols:
    #         self._assign_edge_properties(df_exploded, edge_property_cols, edge_property_mode)

    # def _assign_vertex_properties(self, unique_nodes, from_node_property_cols, to_node_property_cols, df_exploded, from_layer_col, to_layer_col):
    #     # Assign properties for source nodes
    #     if from_node_property_cols:
    #         for prop in from_node_property_cols:
    #             sample_value = df_exploded[prop].dropna().iloc[0]
    #             prop_type = self._infer_property_type(sample_value)
    #             self._ensure_vertex_property(prop, prop_type)

    #             # Group by from_layer_col and 'from_nodes'
    #             prop_values = df_exploded.groupby([from_layer_col, 'from_nodes'])[prop].first()
    #             indices = [self.node_map[(layer, node_id)] for layer, node_id in prop_values.index]
    #             self.graph.vertex_properties[prop].a[indices] = prop_values.values

    #     # Assign properties for target nodes
    #     if to_node_property_cols:
    #         for prop in to_node_property_cols:
    #             sample_value = df_exploded[prop].dropna().iloc[0]
    #             prop_type = self._infer_property_type(sample_value)
    #             self._ensure_vertex_property(prop, prop_type)

    #             # Group by to_layer_col and 'to_nodes'
    #             prop_values = df_exploded.groupby([to_layer_col, 'to_nodes'])[prop].first()
    #             indices = [self.node_map[(layer, node_id)] for layer, node_id in prop_values.index]
    #             self.graph.vertex_properties[prop].a[indices] = prop_values.values

    # def _assign_edge_properties(self, df_exploded, edge_property_cols, edge_property_mode):
    #     for prop in edge_property_cols:
    #         sample_value = df_exploded[prop].dropna().iloc[0]
    #         prop_type = self._infer_property_type(sample_value)
    #         self._ensure_edge_property(prop, prop_type)

    #         # Assign property values
    #         self.graph.edge_properties[prop].a = df_exploded[prop].values

    # def _infer_property_type(self, value):
    #     if isinstance(value, float):
    #         return "float"
    #     elif isinstance(value, int):
    #         return "int"
    #     else:
    #         return "string"

        # Expand the DataFrame to have one row per edge (cartesian product)
        if verbose:
            print("Expanding the DataFrame to generate all edge combinations...")
        df_exploded = df.copy()
        df_exploded['from_nodes_split'] = df_exploded[from_node_id_col].apply(
            lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
        )
        df_exploded['to_nodes_split'] = df_exploded[to_node_id_col].apply(
            lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
        )

        # Sequentially explode 'from_nodes_split' and then 'to_nodes_split'
        df_exploded = df_exploded.explode('from_nodes_split')
        df_exploded = df_exploded.explode('to_nodes_split').reset_index(drop=True)

        # Rename columns for clarity
        df_exploded.rename(columns={
            'from_nodes_split': 'from_nodes',
            'to_nodes_split': 'to_nodes'
        }, inplace=True)

        # Filter out invalid rows
        if verbose:
            print("Filtering out invalid rows...")
        df_exploded = df_exploded.dropna(subset=['from_nodes', 'to_nodes'])
        df_exploded = df_exploded[df_exploded['from_nodes'].apply(lambda x: isinstance(x, str))]
        df_exploded = df_exploded[df_exploded['to_nodes'].apply(lambda x: isinstance(x, str))]

        # Handle null or blank node_ids based on null_node_id parameter
        if verbose:
            print("Handling null or blank node IDs...")
        def handle_null_node_id(node_id):
            if pd.isna(node_id) or node_id == '':
                if null_node_id == 'skip':
                    return None
                elif null_node_id == 'set_to_missing':
                    return 'Missing'
                elif isinstance(null_node_id, str):
                    return null_node_id
                else:
                    return None
            else:
                return node_id

        df_exploded['from_nodes'] = df_exploded['from_nodes'].apply(handle_null_node_id)
        df_exploded['to_nodes'] = df_exploded['to_nodes'].apply(handle_null_node_id)

        # Remove rows where node_id was set to None (i.e., skipped)
        if null_node_id == 'skip':
            df_exploded = df_exploded.dropna(subset=['from_nodes', 'to_nodes'])

        # Create a combined DataFrame of all unique nodes
        if verbose:
            print("Identifying unique nodes from sources and targets...")
        unique_from = df_exploded[['from_layer_col', 'from_nodes']].rename(
            columns={'from_layer_col': 'layer', 'from_nodes': 'node_id'}
        ).drop_duplicates()
        unique_to = df_exploded[['to_layer_col', 'to_nodes']].rename(
            columns={'to_layer_col': 'layer', 'to_nodes': 'node_id'}
        ).drop_duplicates()

        unique_nodes = pd.concat([unique_from, unique_to]).drop_duplicates()

        # Handle missing node_ids based on null_node_id
        if null_node_id in ['set_to_missing'] or isinstance(null_node_id, str):
            unique_nodes['node_id'] = unique_nodes['node_id'].replace('', 'Missing')
            unique_nodes['node_id'] = unique_nodes['node_id'].fillna('Missing')

        # Extract node properties from edge DataFrame
        if verbose:
            print("Extracting node properties from the DataFrame...")
        node_props_source = pd.DataFrame()
        node_props_target = pd.DataFrame()
        if from_node_property_cols:
            node_props_source = df_exploded[['from_layer_col', 'from_nodes'] + from_node_property_cols].copy()
            node_props_source.rename(columns={'from_layer_col': 'layer', 'from_nodes': 'node_id'}, inplace=True)
        if to_node_property_cols:
            node_props_target = df_exploded[['to_layer_col', 'to_nodes'] + to_node_property_cols].copy()
            node_props_target.rename(columns={'to_layer_col': 'layer', 'to_nodes': 'node_id'}, inplace=True)
        node_props = pd.concat([node_props_source, node_props_target]).drop_duplicates()

        # Group by (layer, node_id) and take the first occurrence
        node_props = node_props.groupby(['layer', 'node_id']).first().reset_index()

        # Merge with unique_nodes
        unique_nodes = unique_nodes.merge(node_props, on=['layer', 'node_id'], how='left')

        # Add missing nodes to the graph
        if create_missing and not unique_nodes.empty:
            if verbose:
                print(f"Adding {len(unique_nodes)} unique nodes to the graph...")
            for _, row in unique_nodes.iterrows():
                layer = row['layer'] if pd.notna(row['layer']) else 'Unknown'
                node_id = row['node_id']
                key = (layer, node_id)

                if key in self.node_map:
                    continue  # Node already exists

                # Extract node properties
                properties = {}
                if from_node_property_cols:
                    for prop in from_node_property_cols:
                        properties[prop] = row.get(prop, "Unknown")
                if to_node_property_cols:
                    for prop in to_node_property_cols:
                        properties[prop] = row.get(prop, "Unknown")

                v = self._add_node(layer, node_id, properties)
            if verbose:
                print(f"Added {len(unique_nodes)} nodes to the graph.")
        else:
            if verbose:
                print("create_missing=False. Skipping edges with missing nodes.")

        # Prepare the edge list with (layer, node_id) tuples
        if verbose:
            print("Preparing edge list...")
        df_exploded['from_key'] = list(zip(df_exploded['from_layer_col'], df_exploded['from_nodes']))
        df_exploded['to_key'] = list(zip(df_exploded['to_layer_col'], df_exploded['to_nodes']))

        # Remove edges where from_key or to_key is not in node_map
        if not create_missing:
            df_exploded = df_exploded[
                df_exploded['from_key'].isin(self.node_map) &
                df_exploded['to_key'].isin(self.node_map)
            ]

        # Map to vertex indices
        if verbose:
            print("Mapping node keys to vertex indices...")
        from_vertices = df_exploded['from_key'].map(self.node_map)
        to_vertices = df_exploded['to_key'].map(self.node_map)

        # Check for any remaining missing mappings
        missing_from = from_vertices.isna().sum()
        missing_to = to_vertices.isna().sum()
        if missing_from > 0 or missing_to > 0:
            if verbose:
                print(f"Warning: {missing_from} 'from_nodes' and {missing_to} 'to_nodes' could not be mapped.")
            # Remove rows with missing mappings
            valid_mask = from_vertices.notna() & to_vertices.notna()
            df_exploded = df_exploded[valid_mask].copy()
            from_vertices = from_vertices[valid_mask].astype(int).tolist()
            to_vertices = to_vertices[valid_mask].astype(int).tolist()
        else:
            from_vertices = from_vertices.astype(int).tolist()
            to_vertices = to_vertices.astype(int).tolist()

        # Prepare the edge list
        edge_list = list(zip(from_vertices, to_vertices))
        total_edges = len(edge_list)
        if verbose:
            print(f"Total edges to add: {total_edges}")

        if not edge_list:
            if verbose:
                print("No edges to add. Exiting bulk addition.")
            return

        # Add edges in bulk
        if verbose:
            print("Adding edges in bulk...")
        try:
            added_edges = self.graph.add_edge_list(edge_list)
            if isinstance(added_edges, list):
                print(f"Added {len(added_edges)} edges to the graph.")
            elif added_edges is None:
                # Assuming that edges were added even if add_edge_list returns None
                print(f"Added {len(edge_list)} edges to the graph.")
            else:
                # Handle other possible return types if any
                print(f"Added edges to the graph. Return type of add_edge_list: {type(added_edges)}")
        except Exception as e:
            print(f"Error during add_edge_list: {e}")
            return

        # Assign edge properties
        if edge_property_cols:
            if verbose:
                print("Assigning edge properties in bulk...")
            for prop in edge_property_cols:
                if prop in df_exploded.columns:
                    # Determine property type based on data
                    sample_value = df_exploded[prop].dropna().iloc[0]
                    if isinstance(sample_value, float):
                        prop_type = "float"
                    elif isinstance(sample_value, int):
                        prop_type = "int"
                    else:
                        prop_type = "string"

                    # Ensure the property exists
                    if prop not in self.graph.edge_properties:
                        self.graph.edge_properties[prop] = self.graph.new_edge_property(prop_type)
                    
                    # Assign the property values
                    self.graph.edge_properties[prop].a = df_exploded[prop].values
            if verbose:
                print(f"Assigned edge properties: {edge_property_cols}")

    def print_node_properties(self, node_id, layer):
        """
        Print the properties of a specific node.

        :param node_id: The node ID.
        :param layer: The layer of the node.
        """
        key = (layer, node_id)
        if key in self.node_map:
            v = self.node_map[key]
            properties = {}
            for prop in self.graph.vertex_properties:
                properties[prop] = self.graph.vertex_properties[prop][v]
            print(f"Properties for node {key}:")
            for prop, value in properties.items():
                print(f"  {prop}: {value}")
        else:
            print(f"Node {key} not found in the graph.")

    def visualize_subgraph(self, target_node_id, target_layer, radius=2, filename='subgraph_visualization.png'):
        """
        Visualize a subgraph around the target node.

        :param target_node_id: The node ID around which to visualize the subgraph.
        :param target_layer: The layer of the target node.
        :param radius: The radius (number of hops) around the target node.
        :param filename: Filename to save the visualization image.
        """
        key = (target_layer, target_node_id)
        if key in self.node_map:
            target_vertex = self.node_map[key]
            # Get vertices within the specified radius
            sub_vertices = list(self.graph.get_vertices(radius, [target_vertex]))
            subgraph = self.graph.subgraph(sub_vertices)

            # Define vertex colors based on layer
            layers = subgraph.vertex_properties["layer"]
            unique_layers = list(set(layers.a))
            cmap = plt.get_cmap('tab10')
            layer_to_color = {layer: cmap(i % 10) for i, layer in enumerate(unique_layers)}
            vertex_colors = [layer_to_color.get(layers[v], (0.5, 0.5, 0.5, 1)) for v in subgraph.vertices()]

            # Draw the subgraph
            graph_draw(
                subgraph,
                vertex_fill_color=vertex_colors,
                vertex_text=subgraph.vertex_properties["node_id"],
                output_size=(800, 800),
                vertex_font_size=10,
                edge_pen_width=1.2,
                output=filename
            )
            print(f"Subgraph visualization saved as '{filename}'.")
        else:
            print(f"Node '{target_node_id}' in layer '{target_layer}' not found in the graph.")

    def validate_edge_properties_length(self, edge_properties, num_edges):
        """
        Validates that all lists in edge_properties match the number of edges.
        Raises an error if there is a mismatch.

        :param edge_properties: Dictionary of edge properties.
        :param num_edges: Expected number of edges.
        """
        for prop_name, prop_value in edge_properties.items():
            if isinstance(prop_value, list) and len(prop_value) != num_edges:
                raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match "
                                f"the expected number of edges ({num_edges}).")

    # def get_set_create_node(self, layer, node_id, create_missing=True, verbose=True, **extra_properties):
    #     """
    #     Retrieve an existing node or create a new one if it doesn't exist, then set its properties.

    #     :param layer: Layer name for the node.
    #     :param node_id: Unique identifier for the node.
    #     :param create_missing: If True, create the node if it doesn't exist.
    #     :param verbose: If True, print debug information.
    #     :param extra_properties: Additional properties to set on the node.
    #     :return: The node object if found or created, else None.
    #     """
    #     node_key = (layer, node_id)
    #     node = self.node_map.get(node_key)

    #     if node is None:
    #         if create_missing:
    #             if verbose:
    #                 print(f"Node {node_key} not found. Creating node.")
    #             node = self.graph.add_vertex()
    #             self.node_map[node_key] = node
    #             self._set_node_property(node, "layer", layer)
    #             self._set_node_property(node, "node_id", node_id)
    #         else:
    #             if verbose:
    #                 print(f"Node {node_key} not found and create_missing is False.")
    #             return None

    #     # Set additional properties
    #     for prop_name, prop_value in extra_properties.items():
    #         self._set_node_property(node, prop_name, prop_value)

    #     if verbose:
    #         props = { "layer": layer, "node_id": node_id }
    #         props.update(extra_properties)
    #         print(f"Node {node_id}: {props}")

    #     return node


    # def add_edge(self, from_node, to_node, from_node_id, to_node_id, from_layer, to_layer, create_missing=True, skip_if_duplicate=None, verbose=True, **edge_properties):
    #     """
    #     Add a directed edge between two nodes with specified properties, with optional duplicate handling.

    #     :param from_node: The source node (vertex) of the edge.
    #     :param to_node: The target node (vertex) of the edge.
    #     :param from_node_id: The ID of the source node.
    #     :param to_node_id: The ID of the target node.
    #     :param from_layer: The layer of the source node.
    #     :param to_layer: The layer of the target node.
    #     :param create_missing: If False, raises an error if either node does not exist.
    #     :param skip_if_duplicate: If None, allows duplicate edges; if 'any', skips if any edge exists;
    #                             if 'exact', skips only if an edge with identical properties exists.
    #     :param edge_properties: Additional properties to apply to the edge.
    #     :param verbose: If True, prints debugging information.
    #     :return: The created edge object, or None if the edge was skipped.

    #     Raises:
    #     - RuntimeError: If `create_missing` is False and a node does not exist in the node map.
    #     """
    #     # Check node existence if create_missing is False
    #     if not create_missing:
    #         if from_node is None:
    #             raise RuntimeError(f"Node '{from_node_id}' in layer '{from_layer}' does not exist. "
    #                             f"Set `create_missing=True` to create missing nodes.")
    #         if to_node is None:
    #             raise RuntimeError(f"Node '{to_node_id}' in layer '{to_layer}' does not exist. "
    #                             f"Set `create_missing=True` to create missing nodes.")

    #     # Handle duplicate edge behavior based on skip_if_duplicate
    #     existing_edges = self.graph.edge(from_node, to_node, all_edges=True)
    #     if existing_edges:
    #         if skip_if_duplicate == "any":
    #             # Skip if any edge exists between the nodes
    #             if verbose:
    #                 print(f"Edge from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) already exists. Skipping due to 'any' duplicate policy.")
    #             return None

    #         elif skip_if_duplicate == "exact":
    #             # Check if an exact edge with identical properties exists
    #             for edge in existing_edges:
    #                 identical = all(
    #                     (prop in self.graph.edge_properties and self.graph.edge_properties[prop][edge] == value)
    #                     for prop, value in edge_properties.items()
    #                 )
    #                 if identical:
    #                     if verbose:
    #                         print(f"Identical edge from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) exists. Skipping due to 'exact' duplicate policy.")
    #                     return None

    #     # Create the edge between nodes
    #     edge = self.graph.add_edge(from_node, to_node)

    #     # Set edge properties using _set_edge_property
    #     for prop_name, prop_value in edge_properties.items():
    #         self._set_edge_property(edge, prop_name, prop_value)

    #     if verbose:
    #         print(f"Edge added from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) with properties: {edge_properties}")

    #     return edge

    
    # def add_edges_from_pairs(self, node_pairs, split_char='|', create_missing=False, skip_if_duplicate='exact', **edge_properties):
    #     """
    #     Add directed edges between pairs of nodes using node_pairs, with optional per-edge properties.

    #     Args:
    #     - node_pairs: A list of tuples (node1_id, node2_id), where each node is identified by (layer, node_id).
    #     - split_char: Character to split multiple node IDs.
    #     - create_missing: If True, creates missing nodes if they are not found in node_map.
    #     - edge_properties: Additional properties for each created edge. Properties can be single values or lists.
    #     """
    #     # Calculate expected number of edges
    #     expected_num_edges = sum(
    #         len([id_.strip() for id_ in node1_id[1].split(split_char)]) *
    #         len([id_.strip() for id_ in node2_id[1].split(split_char)])
    #         for node1_id, node2_id in node_pairs if node1_id and node2_id
    #     )

    #     # Validate list lengths
    #     self.validate_edge_properties_length(edge_properties, expected_num_edges)

    #     edge_count = 0  # Counter for each edge added, used as index for property lists

    #     for node1_id, node2_id in node_pairs:
    #         if node1_id is None or node2_id is None:
    #             continue
            
    #         node1_ids = [id_.strip() for id_ in node1_id[1].split(split_char)] if split_char in node1_id[1] else [node1_id[1]]
    #         node2_ids = [id_.strip() for id_ in node2_id[1].split(split_char)] if split_char in node2_id[1] else [node2_id[1]]

    #         for id1 in node1_ids:
    #             for id2 in node2_ids:
    #                 from_layer = node1_id[0]
    #                 to_layer = node2_id[0]
    #                 node1 = self.get_set_create_node(layer=from_layer, node_id=id1, create_missing=create_missing)
    #                 node2 = self.get_set_create_node(layer=to_layer, node_id=id2, create_missing=create_missing)

    #                 if node1 is None or node2 is None:
    #                     continue
                    
    #                 # Resolve edge-specific properties using helper function
    #                 edge_specific_properties = self.resolve_edge_properties(edge_properties, edge_count)
                    
    #                 # Use add_edge to create a single edge with resolved properties
    #                 self.add_edge(
    #                     node1, node2,
    #                     from_node_id=node1_ids, to_node_id=node2_ids,
    #                     from_layer=from_layer, to_layer=to_layer,
    #                     create_missing=create_missing,
    #                     skip_if_duplicate=skip_if_duplicate,
    #                     **edge_specific_properties
    #                 )
    #                 edge_count += 1


    # def add_edges_from_nodes(
    #     self,
    #     from_nodes,
    #     to_nodes,
    #     from_layer,
    #     to_layer,
    #     split_char='|',
    #     create_missing=False,
    #     skip_if_duplicate='exact',
    #     verbose=True,
    #     **edge_properties
    #     ):
    #         """
    #         Adds directed edges from 'from_nodes' to 'to_nodes' using specified layers and optional per-edge properties.
    #         Validates that any list properties match the expected number of edges.

    #         Args:
    #         - from_nodes: List or iterable of node IDs for 'from' nodes.
    #         - to_nodes: List or iterable of node IDs for 'to' nodes.
    #         - from_layer: Layer name for all 'from' nodes.
    #         - to_layer: Layer name for all 'to' nodes.
    #         - split_char: Character to split multiple node IDs.
    #         - create_missing: If True, creates missing nodes if they are not found in node_map.
    #         - skip_if_duplicate: If 'exact', skips adding duplicate edges based on exact matches. Other options can be implemented as needed.
    #         - verbose: If True, prints debug information.
    #         - edge_properties: Additional properties for each created edge. Properties can be single values or lists.

    #         Raises:
    #         - ValueError: If any list property in edge_properties does not match the expected number of edges.
    #         """
    #         if len(from_nodes) != len(to_nodes):
    #             raise ValueError("from_nodes and to_nodes must have the same length.")

    #         # Initialize counters
    #         expected_num_edges = 0
    #         skipped_rows_initial = 0
    #         skipped_rows = []

    #         # First pass: Calculate the expected number of edges, safely handling non-string node IDs
    #         for from_node_id, to_node_id in zip(from_nodes, to_nodes):
    #             if isinstance(from_node_id, str) and isinstance(to_node_id, str) and from_node_id and to_node_id:
    #                 from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)]
    #                 to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)]
    #                 expected_num_edges += len(from_node_ids) * len(to_node_ids)
    #             else:
    #                 skipped_rows_initial += 1
    #                 skipped_rows.append(f"Skipped: From node: {from_node_id}, To node: {from_node_id}")

    #         if verbose and skipped_rows_initial > 0:
    #             print(f"Skipped {skipped_rows_initial} rows due to non-string node IDs or missing values.\n{skipped_rows}")

    #         # Validate list lengths in edge_properties
    #         self.validate_edge_properties_length(edge_properties, expected_num_edges)

    #         edge_count = 0  # Counter for each edge added
    #         skipped_rows = 0  # Counter for skipped rows during edge addition

    #         # Second pass: Iterate through from_nodes and to_nodes to add edges
    #         for idx, (from_node_id, to_node_id) in enumerate(zip(from_nodes, to_nodes)):
    #             # Check if both node IDs are strings
    #             if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
    #                 if verbose:
    #                     print(f"Skipping row {idx} due to non-string node IDs: from_node_id={from_node_id}, to_node_id={to_node_id}")
    #                 skipped_rows += 1
    #                 continue

    #             # Split node IDs by split_char and strip whitespace
    #             from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
    #             to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

    #             # Iterate through all combinations of from_node_ids and to_node_ids
    #             for id1 in from_node_ids:
    #                 for id2 in to_node_ids:
    #                     # Retrieve or create nodes
    #                     node1 = self.get_set_create_node(
    #                         layer=from_layer,
    #                         node_id=id1,
    #                         create_missing=create_missing,
    #                         verbose=verbose
    #                         # **edge_properties
    #                     )
    #                     node2 = self.get_set_create_node(
    #                         layer=to_layer,
    #                         node_id=id2,
    #                         create_missing=create_missing,
    #                         verbose=verbose
    #                         # **edge_properties
    #                     )

    #                     # Skip if either node couldn't be retrieved or created
    #                     if node1 is None or node2 is None:
    #                         if verbose:
    #                             print(f"Skipping edge from '{id1}' to '{id2}' because one of the nodes could not be retrieved or created.")
    #                         continue

    #                     # Resolve edge-specific properties using helper function
    #                     edge_specific_properties = self.resolve_edge_properties(edge_properties, edge_count)

    #                     # Add the edge with resolved properties
    #                     self.add_edge(
    #                         node1, node2,
    #                         from_node_id=from_node_id,
    #                         to_node_id=to_node_id,
    #                         from_layer=from_layer,
    #                         to_layer=to_layer,
    #                         create_missing=create_missing,
    #                         skip_if_duplicate=skip_if_duplicate,
    #                         verbose=verbose,
    #                         **edge_specific_properties
    #                     )

    #                     if verbose:
    #                         print(f"Added edge from '{id1}' to '{id2}' with properties {edge_specific_properties}")

    #                     edge_count += 1

    #         if verbose:
    #             print(f"Total edges expected to add: {expected_num_edges}")
    #             print(f"Total edges actually added: {edge_count}")
    #             if skipped_rows > 0:
    #                 print(f"Total rows skipped during edge addition: {skipped_rows}")


    # def add_edges_from_dataframe(
    #     self,
    #     df: pd.DataFrame,
    #     from_col: str,
    #     to_col: str,
    #     from_layer: str = None,
    #     to_layer: str = None,
    #     from_layer_col: str = None,
    #     to_layer_col: str = None,
    #     edge_property_cols: list = None,
    #     from_node_property_cols: list = None,
    #     to_node_property_cols: list = None,
    #     split_char: str = '|',
    #     create_missing: bool = False,
    #     skip_if_duplicate: str = 'exact',
    #     verbose: bool = True,
    #     edge_property_mode: str = 'per-edge'  # New parameter
    # ):
    #     """
    #     Adds directed edges from a DataFrame, using specified columns for node IDs and optional layers.
    #     Allows customizable edge properties by specifying which DataFrame columns to use.
    #     Additionally, allows setting or updating custom properties of source and target nodes
    #     based on specified DataFrame columns.
    #     Supports multiple IDs in a single cell, separated by a specified character.

    #     Args:
    #     - df (pd.DataFrame): DataFrame containing edge and node information.
    #     - from_col (str): Column name in df for source node IDs.
    #     - to_col (str): Column name in df for target node IDs.
    #     - from_layer (str, optional): Fixed layer name for all source nodes if `from_layer_col` is not specified.
    #     - to_layer (str, optional): Fixed layer name for all target nodes if `to_layer_col` is not specified.
    #     - from_layer_col (str, optional): Column in df specifying 'from' node layers if layers vary by row.
    #     - to_layer_col (str, optional): Column in df specifying 'to' node layers if layers vary by row.
    #     - edge_property_cols (list of str, optional): List of column names in df to be used as edge properties.
    #     - from_node_property_cols (list of str, optional): List of column names in df to be used as properties for source nodes.
    #     - to_node_property_cols (list of str, optional): List of column names in df to be used as properties for target nodes.
    #     - split_char (str, optional): Character used to split multiple node IDs within a single cell. Default is '|'.
    #     - create_missing (bool, optional): If True, creates missing nodes when they are not found in node_map. Default is False.
    #     - skip_if_duplicate (str, optional): Strategy to handle duplicate edges ('exact', 'any', or None). Default is 'exact'.
    #     - verbose (bool, optional): If True, prints debug information. Default is True.
    #     - edge_property_mode (str, optional): Mode for handling edge properties.
    #         - 'per-edge': Each edge has its own property value. Requires property lists to match the number of edges.
    #         - 'per-row': Each row provides a single property value applied to all edges generated from that row.
    #         Default is 'per-edge'.

    #     Raises:
    #     - ValueError: If no fixed layer name or column name is provided for either source or target layers.
    #     - ValueError: If edge_property_mode is not one of the accepted values.
    #     """
    #     # Validate edge_property_mode
    #     if edge_property_mode not in ['per-edge', 'per-row']:
    #         raise ValueError("`edge_property_mode` must be either 'per-edge' or 'per-row'.")

    #     # Validate input DataFrame
    #     if not isinstance(df, pd.DataFrame):
    #         raise TypeError("df must be a pandas DataFrame.")

    #     # Determine edge properties to use
    #     excluded_cols = {from_col, to_col}
    #     if from_layer_col:
    #         excluded_cols.add(from_layer_col)
    #     if to_layer_col:
    #         excluded_cols.add(to_layer_col)
    #     if edge_property_cols:
    #         excluded_cols.update(edge_property_cols)
    #     if from_node_property_cols:
    #         excluded_cols.update(from_node_property_cols)
    #     if to_node_property_cols:
    #         excluded_cols.update(to_node_property_cols)

    #     # If edge_property_cols is not specified, exclude node-related columns
    #     if edge_property_cols is None:
    #         property_cols = [col for col in df.columns if col not in excluded_cols]
    #     else:
    #         property_cols = edge_property_cols

    #     # Prepare edge properties by extracting specified columns as lists
    #     edge_properties = {}
    #     for col in property_cols:
    #         if col in df.columns:
    #             edge_properties[col] = df[col].tolist()
    #         else:
    #             raise ValueError(f"Edge property column '{col}' not found in DataFrame.")

    #     # Prepare node property columns
    #     from_node_properties = from_node_property_cols if from_node_property_cols else []
    #     to_node_properties = to_node_property_cols if to_node_property_cols else []

    #     # Ensure that either a fixed layer or a column is provided for source and target layers
    #     if from_layer is None and from_layer_col is None:
    #         raise ValueError("Either `from_layer` or `from_layer_col` must be specified for source node layers.")
    #     if to_layer is None and to_layer_col is None:
    #         raise ValueError("Either `to_layer` or `to_layer_col` must be specified for target node layers.")

    #     # Calculate the expected number of edges for property validation
    #     expected_num_edges = 0
    #     skipped_due_to_invalid = 0

    #     for idx, row in df.iterrows():
    #         from_node_id = row[from_col]
    #         to_node_id = row[to_col]

    #         if pd.isna(from_node_id) or pd.isna(to_node_id):
    #             skipped_due_to_invalid += 1
    #             continue

    #         if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
    #             skipped_due_to_invalid += 1
    #             continue

    #         from_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
    #         to_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

    #         expected_num_edges += len(from_ids) * len(to_ids)

    #     # Validate list lengths in edge_properties based on mode
    #     if edge_property_mode == 'per-edge':
    #         # Each property list must match the number of edges
    #         self.validate_edge_properties_length(edge_properties, expected_num_edges)
    #     elif edge_property_mode == 'per-row':
    #         # Each property list must match the number of rows (excluding skipped)
    #         num_valid_rows = len(df) - skipped_due_to_invalid
    #         for prop_name, prop_value in edge_properties.items():
    #             if isinstance(prop_value, list):
    #                 if len(prop_value) != len(df):
    #                     raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match the number of rows ({len(df)}).")
    #     # Note: 'per-row' mode assumes that the property value for a row applies to all edges generated from that row.

    #     # Initialize counters and trackers
    #     edge_count = 0  # Counter for each edge added, used as index for property lists
    #     total_edges_added = 0

    #     # Iterate through each row to add edges
    #     for idx, row in df.iterrows():
    #         from_node_id = row[from_col]
    #         to_node_id = row[to_col]

    #         if pd.isna(from_node_id) or pd.isna(to_node_id):
    #             if verbose:
    #                 if pd.isna(from_node_id) and pd.isna(to_node_id):
    #                     print(f"Skipping row {idx}: Both 'from_col' and 'to_col' are NaN. Row data: {row.to_dict()}")
    #                 elif pd.isna(from_node_id):
    #                     print(f"Skipping row {idx}: 'from_col' is NaN. Row data: {row.to_dict()}")
    #                 elif pd.isna(to_node_id):
    #                     print(f"Skipping row {idx}: 'to_col' is NaN. Row data: {row.to_dict()}")
    #             skipped_due_to_invalid += 1
    #             continue

    #         if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
    #             if verbose:
    #                 issues = []
    #                 if not isinstance(from_node_id, str):
    #                     issues.append(f"'from_col' is of type {type(from_node_id).__name__} with value {from_node_id}")
    #                 if not isinstance(to_node_id, str):
    #                     issues.append(f"'to_col' is of type {type(to_node_id).__name__} with value {to_node_id}")
    #                 issue_details = "; ".join(issues)
    #                 print(f"Skipping row {idx}: {issue_details}. Row data: {row.to_dict()}")
    #             skipped_due_to_invalid += 1
    #             continue

    #         # Split node IDs if necessary
    #         from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
    #         to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

    #         # Determine effective layers
    #         effective_from_layer = from_layer if from_layer else row.get(from_layer_col, from_layer)
    #         effective_to_layer = to_layer if to_layer else row.get(to_layer_col, to_layer)

    #         # Extract node properties from the row
    #         from_node_props = {prop: row[prop] for prop in from_node_properties if prop in row and not pd.isna(row[prop])}
    #         to_node_props = {prop: row[prop] for prop in to_node_properties if prop in row and not pd.isna(row[prop])}

    #         # Resolve edge-specific properties based on mode
    #         if edge_property_mode == 'per-edge':
    #             # For per-edge mode, properties are indexed by edge_count
    #             edge_specific_properties = {}
    #             for prop in edge_properties:
    #                 prop_value = edge_properties[prop]
    #                 if isinstance(prop_value, list):
    #                     edge_specific_properties[prop] = prop_value[edge_count]
    #                 else:
    #                     edge_specific_properties[prop] = prop_value
    #         elif edge_property_mode == 'per-row':
    #             # For per-row mode, properties are indexed by row index
    #             edge_specific_properties = {}
    #             for prop in edge_properties:
    #                 prop_value = edge_properties[prop]
    #                 if isinstance(prop_value, list):
    #                     edge_specific_properties[prop] = prop_value[idx]
    #                 else:
    #                     edge_specific_properties[prop] = prop_value

    #         # Iterate through all combinations of from_node_ids and to_node_ids
    #         for id1 in from_node_ids:
    #             for id2 in to_node_ids:
    #                 # Retrieve or create source node with custom properties
    #                 node1 = self.get_set_create_node(
    #                     layer=effective_from_layer,
    #                     node_id=id1,
    #                     create_missing=create_missing,
    #                     verbose=verbose,
    #                     **from_node_props
    #                 )

    #                 # Retrieve or create target node with custom properties
    #                 node2 = self.get_set_create_node(
    #                     layer=effective_to_layer,
    #                     node_id=id2,
    #                     create_missing=create_missing,
    #                     verbose=verbose,
    #                     **to_node_props
    #                 )

    #                 if node1 is None or node2 is None:
    #                     if verbose:
    #                         print(f"Skipping edge from '{id1}' to '{id2}': Missing nodes.")
    #                     continue

    #                 # Add the edge with resolved properties
    #                 self.add_edge(
    #                     from_node=node1,
    #                     to_node=node2,
    #                     from_node_id=id1,
    #                     to_node_id=id2,
    #                     from_layer=effective_from_layer,
    #                     to_layer=effective_to_layer,
    #                     create_missing=create_missing,
    #                     skip_if_duplicate=skip_if_duplicate,
    #                     verbose=verbose,
    #                     **edge_specific_properties
    #                 )
    #                 edge_count += 1
    #                 total_edges_added += 1

    #     # Summary of the operation
    #     if verbose:
    #         print(f"Total edges expected to be added: {expected_num_edges}")
    #         print(f"Total edges actually added: {total_edges_added}")
    #         if skipped_due_to_invalid > 0:
    #             print(f"Total rows skipped due to invalid node IDs: {skipped_due_to_invalid}")


    # def resolve_edge_properties(self, edge_properties, edge_index):
    #     """
    #     Resolve edge properties for a specific edge, handling both single values and lists.

    #     :param edge_properties: Dictionary of edge properties, where values can be lists or single values.
    #     :param edge_index: The index of the current edge being processed.
    #     :return: A dictionary of properties specific to the current edge.
    #     """
    #     resolved_properties = {}
    #     for prop_name, prop_value in edge_properties.items():
    #         if isinstance(prop_value, list):
    #             resolved_properties[prop_name] = prop_value[edge_index]
    #         else:
    #             resolved_properties[prop_name] = prop_value
    #     return resolved_properties


    # def validate_edge_properties_length(self, edge_properties, num_edges):
    #     """
    #     Validates that all lists in edge_properties match the number of edges.
    #     Raises an error if there is a mismatch.

    #     :param edge_properties: Dictionary of edge properties.
    #     :param num_edges: Expected number of edges.
    #     """
    #     for prop_name, prop_value in edge_properties.items():
    #         if isinstance(prop_value, list) and len(prop_value) != num_edges:
    #             raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match "
    #                             f"the expected number of edges ({num_edges}).")


    # def _set_node_property(self, node, prop_name, value, nan_replacement=None):
    #     """
    #     Set or create a node property for a given node.
        
    #     :param node: The node (vertex) to assign the property.
    #     :param prop_name: The name of the property.
    #     :param value: The value of the property, which determines the property type.
    #     :param nan_replacement: Optional; the value to replace NaN with (default is None).
    #     """
    #     import math
    #     # Replace NaN with the specified replacement, if value is NaN
    #     if isinstance(value, float) and math.isnan(value):
    #         value = nan_replacement

    #     # Check if the property map exists, create it if not
    #     if prop_name not in self.graph.vp:
    #         # Determine type based on the value type for simplicity
    #         if isinstance(value, str):
    #             prop_type = "string"
    #         elif isinstance(value, float):
    #             prop_type = "float"
    #         elif isinstance(value, int):
    #             prop_type = "int"
    #         else:
    #             prop_type = "object"  # For more complex data types

    #         self.graph.vp[prop_name] = self.graph.new_vertex_property(prop_type)
        
    #     # Set the property value for the node
    #     self.graph.vp[prop_name][node] = value


    # def _set_edge_property(self, edge, prop_name, value, nan_replacement=None):
    #     """
    #     Set or create an edge property for a given edge.
        
    #     :param edge: The edge to assign the property.
    #     :param prop_name: The name of the property.
    #     :param value: The value of the property, which determines the property type.
    #     :param nan_replacement: Optional; the value to replace NaN with (default is None).
    #     """
    #     import math
    #     # Replace NaN with the specified replacement, if value is NaN
    #     if isinstance(value, float) and math.isnan(value):
    #         value = nan_replacement

    #     # Check if the property map exists, create it if not
    #     if prop_name not in self.graph.ep:
    #         # Determine type based on the value type
    #         if isinstance(value, str):
    #             prop_type = "string"
    #         elif isinstance(value, float):
    #             prop_type = "float"
    #         elif isinstance(value, int):
    #             prop_type = "int"
    #         else:
    #             prop_type = "object"  # For more complex data types

    #         self.graph.ep[prop_name] = self.graph.new_edge_property(prop_type)
        
    #     # Set the property value for the edge
    #     self.graph.ep[prop_name][edge] = value


    # def build_layer(self, nodes, layer_name, custom_node_properties={}, verbose=True):
    #     """
    #     Build a network layer from a dataset.
        
    #     :param layer_data: List of desired node ids
    #     :param layer_name: Name of the layer
    #     :param custom_properties: Dictionary of extra properties for each node {node_id: {property_name: value}}
    #     :param verbose: Whether to print every node update.
    #     """
    #     custom_node_properties = custom_node_properties or {}
    #     for node_id in nodes:
    #         # Get extra properties for this node if they exist
    #         individual_node_properties = custom_node_properties.get(node_id, {})
    #         self.get_set_create_node(layer=layer_name, node_id=node_id, verbose=verbose, create_missing=True, **individual_node_properties)
    
    
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



# import pandas as pd
# import numpy as np
# from graph_tool.all import Graph, graph_draw
# import math
# import matplotlib.pyplot as plt  # For visualization


class MultilayerNetworkGraphTool:
    def __init__(self):
        self.graph = Graph(directed=True)
        self.node_map = {}  # Maps (layer, node_id) to vertex

        # Define vertex properties
        self.vp_node_id = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["node_id"] = self.vp_node_id

        self.vp_layer = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["layer"] = self.vp_layer

        # # Define additional node properties (Prop1 to Prop10)
        # for i in range(1, 11):
        #     prop = f'Prop{i}'
        #     self.graph.vertex_properties[prop] = self.graph.new_vertex_property("string")

        # # Define edge properties (EdgeProp1 to EdgeProp10)
        # for i in range(1, 11):
        #     prop = f'EdgeProp{i}'
        #     self.graph.edge_properties[prop] = self.graph.new_edge_property("float")

    def _set_node_property(self, node, prop_name, value, nan_replacement=None):
        """
        Set or create a node property for a given node.

        :param node: The node (vertex) to assign the property.
        :param prop_name: The name of the property.
        :param value: The value of the property, which determines the property type.
        :param nan_replacement: Optional; the value to replace NaN with (default is None).
        """
        # Replace NaN with the specified replacement, if value is NaN
        if isinstance(value, float) and math.isnan(value):
            value = nan_replacement

        # Determine the property map type based on the value
        if prop_name not in self.graph.vertex_properties:
            if isinstance(value, str):
                prop_type = "string"
            elif isinstance(value, float):
                prop_type = "float"
            elif isinstance(value, int):
                prop_type = "int"
            else:
                prop_type = "string"  # Default to string for unknown types

            self.graph.vertex_properties[prop_name] = self.graph.new_vertex_property(prop_type)

        # Set the property value for the node
        self.graph.vertex_properties[prop_name][node] = value

    def _set_edge_property(self, edge, prop_name, value, nan_replacement=None):
        """
        Set or create an edge property for a given edge.

        :param edge: The edge to assign the property.
        :param prop_name: The name of the property.
        :param value: The value of the property, which determines the property type.
        :param nan_replacement: Optional; the value to replace NaN with (default is None).
        """
        # Replace NaN with the specified replacement, if value is NaN
        if isinstance(value, float) and math.isnan(value):
            value = nan_replacement

        # Determine the property map type based on the value
        if prop_name not in self.graph.edge_properties:
            if isinstance(value, str):
                prop_type = "string"
            elif isinstance(value, float):
                prop_type = "float"
            elif isinstance(value, int):
                prop_type = "int"
            else:
                prop_type = "float"  # Default to float for unknown types

            self.graph.edge_properties[prop_name] = self.graph.new_edge_property(prop_type)

        # Set the property value for the edge
        self.graph.edge_properties[prop_name][edge] = value

    def add_edges_from_dataframe(
        self,
        df: pd.DataFrame,
        from_col: str,
        to_col: str,
        from_layer_col: str = None,
        to_layer_col: str = None,
        edge_property_cols: list = None,
        split_char: str = '|',
        create_missing: bool = False,
        skip_if_duplicate: str = 'exact',
        verbose: bool = True,
        edge_property_mode: str = 'per-edge',  # 'per-edge' or 'per-row'
        null_node_id: str = 'skip'  # Options: 'skip', 'set_to_missing', or custom string
    ):
        """
        Adds directed edges from a DataFrame, using specified columns for node IDs and optional layers.
        Automatically handles node creation and null node IDs.

        Args:
        - df (pd.DataFrame): DataFrame containing edge and node information.
        - from_col (str): Column name in df for source node IDs.
        - to_col (str): Column name in df for target node IDs.
        - from_layer_col (str, optional): Column in df specifying source node layers.
        - to_layer_col (str, optional): Column in df specifying target node layers.
        - edge_property_cols (list of str, optional): List of column names in df to be used as edge properties.
        - split_char (str, optional): Character used to split multiple node IDs within a single cell. Default is '|'.
        - create_missing (bool, optional): If True, creates missing nodes when they are not found in node_map. Default is False.
        - skip_if_duplicate (str, optional): Strategy to handle duplicate edges ('exact', 'any', or None). Default is 'exact'.
        - verbose (bool, optional): If True, prints debug information. Default is True.
        - edge_property_mode (str, optional): Mode for handling edge properties.
            - 'per-edge': Each edge has its own property value. Requires property lists to match the number of edges.
            - 'per-row': Each row provides a single property value applied to all edges generated from that row.
            Default is 'per-edge'.
        - null_node_id (str, optional): Strategy for handling null or blank node IDs.
            - 'skip': Exclude nodes with null or blank 'node_id'.
            - 'set_to_missing': Assign a default label like 'Missing'.
            - Custom string: Assign a user-defined label (e.g., 'Unknown').
            Default is 'skip'.
        
        Raises:
        - ValueError: If edge_property_mode is invalid or required layer columns are missing.
        """
        # Validate edge_property_mode
        if edge_property_mode not in ['per-edge', 'per-row']:
            raise ValueError("`edge_property_mode` must be either 'per-edge' or 'per-row'.")

        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        # Determine edge properties to use
        excluded_cols = {from_col, to_col}
        if from_layer_col:
            excluded_cols.add(from_layer_col)
        if to_layer_col:
            excluded_cols.add(to_layer_col)
        if edge_property_cols:
            excluded_cols.update(edge_property_cols)

        # If edge_property_cols is not specified, exclude node-related columns
        if edge_property_cols is None:
            property_cols = [col for col in df.columns if col not in excluded_cols]
        else:
            property_cols = edge_property_cols

        # Prepare edge properties by extracting specified columns as lists
        edge_properties = {}
        for col in property_cols:
            if col in df.columns:
                edge_properties[col] = df[col].tolist()
            else:
                raise ValueError(f"Edge property column '{col}' not found in DataFrame.")

        # Handle edge_property_mode
        if edge_property_mode == 'per-edge':
            # Calculate the total number of edges
            df['from_nodes_split'] = df[from_col].apply(
                lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
            )
            df['to_nodes_split'] = df[to_col].apply(
                lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
            )
            df['num_edges'] = df.apply(lambda row: len(row['from_nodes_split']) * len(row['to_nodes_split']), axis=1)
            expected_num_edges = df['num_edges'].sum()

            # Validate that each edge property list matches the total number of edges
            self.validate_edge_properties_length(edge_properties, expected_num_edges)

        elif edge_property_mode == 'per-row':
            # For 'per-row', ensure that properties are either single values or lists matching the number of rows
            num_valid_rows = len(df.dropna(subset=[from_col, to_col]))
            for prop_name, prop_value in edge_properties.items():
                if isinstance(prop_value, list):
                    if len(prop_value) != len(df):
                        raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match the number of rows ({len(df)}).")
                else:
                    # If single value, replicate it for all rows
                    edge_properties[prop_name] = [prop_value] * len(df)

        # Expand the DataFrame to have one row per edge (cartesian product)
        print("Expanding the DataFrame to generate all edge combinations...")
        df_exploded = df.copy()
        df_exploded['from_nodes_split'] = df_exploded[from_col].apply(
            lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
        )
        df_exploded['to_nodes_split'] = df_exploded[to_col].apply(
            lambda x: [id_.strip() for id_ in x.split(split_char)] if isinstance(x, str) else []
        )

        # Sequentially explode 'from_nodes_split' and then 'to_nodes_split'
        df_exploded = df_exploded.explode('from_nodes_split')
        df_exploded = df_exploded.explode('to_nodes_split').reset_index(drop=True)

        # Rename columns for clarity
        df_exploded.rename(columns={'from_nodes_split': 'from_nodes', 'to_nodes_split': 'to_nodes'}, inplace=True)

        # Filter out invalid rows
        print("Filtering out invalid rows...")
        df_exploded = df_exploded.dropna(subset=['from_nodes', 'to_nodes'])
        df_exploded = df_exploded[df_exploded['from_nodes'].apply(lambda x: isinstance(x, str))]
        df_exploded = df_exploded[df_exploded['to_nodes'].apply(lambda x: isinstance(x, str))]

        # Identify all unique (layer, node_id) pairs in the exploded DataFrame
        df_exploded['from_layer'] = df_exploded[from_layer_col] if from_layer_col else from_layer_col
        df_exploded['to_layer'] = df_exploded[to_layer_col] if to_layer_col else to_layer_col

        unique_from = df_exploded[['from_layer', 'from_nodes']].drop_duplicates()
        unique_to = df_exploded[['to_layer', 'to_nodes']].drop_duplicates()

        # Handle null or blank node_ids based on null_node_id parameter
        def handle_null_node_id(row, node_col):
            node_id = row[node_col]
            layer_col = 'from_layer' if node_col == 'from_nodes' else 'to_layer'
            layer = row[layer_col]
            if pd.isna(node_id) or node_id == '':
                if null_node_id == 'skip':
                    return None
                elif null_node_id == 'set_to_missing':
                    return 'Missing'
                elif isinstance(null_node_id, str):
                    return null_node_id
                else:
                    return None
            else:
                return node_id

        # Apply handling for from_nodes
        unique_from['from_nodes'] = unique_from.apply(lambda row: handle_null_node_id(row, 'from_nodes'), axis=1)
        # Apply handling for to_nodes
        unique_to['to_nodes'] = unique_to.apply(lambda row: handle_null_node_id(row, 'to_nodes'), axis=1)

        # Remove rows where node_id was set to None (i.e., skipped)
        if null_node_id == 'skip':
            unique_from = unique_from.dropna(subset=['from_nodes'])
            unique_to = unique_to.dropna(subset=['to_nodes'])

        # Create a combined DataFrame of all unique nodes
        unique_nodes = pd.concat([
            unique_from.rename(columns={'from_layer': 'layer', 'from_nodes': 'node_id'}),
            unique_to.rename(columns={'to_layer': 'layer', 'to_nodes': 'node_id'})
        ]).drop_duplicates()

        # Add missing nodes to the graph
        if create_missing:
            print(f"Adding {len(unique_nodes)} unique nodes to the graph...")
            for _, row in unique_nodes.iterrows():
                layer = row['layer']
                node_id = row['node_id']
                key = (layer, node_id)

                if key in self.node_map:
                    continue  # Node already exists

                v = self.graph.add_vertex()
                self.vp_node_id[v] = node_id
                self.vp_layer[v] = layer

                # Assign default properties
                for i in range(1, 11):
                    prop = f'Prop{i}'
                    self._set_node_property(v, prop, "Unknown")
                self.node_map[key] = v
            print(f"Added {len(unique_nodes)} nodes to the graph.")
        else:
            print("create_missing=False. Skipping edges with missing nodes.")

        # Prepare the edge list with (layer, node_id) tuples
        df_exploded['from_key'] = list(zip(df_exploded['from_layer'], df_exploded['from_nodes']))
        df_exploded['to_key'] = list(zip(df_exploded['to_layer'], df_exploded['to_nodes']))

        # Remove edges where from_key or to_key is not in node_map
        if not create_missing:
            df_exploded = df_exploded[
                df_exploded['from_key'].isin(self.node_map) &
                df_exploded['to_key'].isin(self.node_map)
            ]

        # Map to vertex indices
        from_vertices = df_exploded['from_key'].map(self.node_map)
        to_vertices = df_exploded['to_key'].map(self.node_map)

        # Check for any remaining missing mappings
        missing_from = from_vertices.isna().sum()
        missing_to = to_vertices.isna().sum()
        if missing_from > 0 or missing_to > 0:
            print(f"Warning: {missing_from} 'from_nodes' and {missing_to} 'to_nodes' could not be mapped.")
            # Remove rows with missing mappings
            valid_mask = from_vertices.notna() & to_vertices.notna()
            df_exploded = df_exploded[valid_mask].copy()
            from_vertices = from_vertices[valid_mask].astype(int).tolist()
            to_vertices = to_vertices[valid_mask].astype(int).tolist()
        else:
            from_vertices = from_vertices.astype(int).tolist()
            to_vertices = to_vertices.astype(int).tolist()

        # Prepare the edge list
        edge_list = list(zip(from_vertices, to_vertices))
        total_edges = len(edge_list)
        print(f"Preparing to add {total_edges} edges...")

        if not edge_list:
            print("No edges to add. Exiting bulk addition.")
            return

        # Add edges in bulk
        print("Adding edges in bulk...")
        self.add_edges_bulk(edge_list, edge_property_cols, df_exploded, edge_property_cols)
        print("Edge addition complete.")

    def add_edges_bulk(self, edge_list, edge_property_cols, df_exploded, edge_property_cols_input):
        """
        Add edges in bulk to the graph using a prepared edge list and assign edge properties.

        :param edge_list: List of tuples representing edges (from_vertex, to_vertex).
        :param edge_property_cols: List of column names in df_exploded to be used as edge properties.
        :param df_exploded: DataFrame containing the exploded edge data.
        :param edge_property_cols_input: Original list of edge property columns.
        """
        try:
            # Add edges in bulk
            added_edges = self.graph.add_edge_list(edge_list)
            if isinstance(added_edges, list):
                print(f"Added {len(added_edges)} edges to the graph.")
            elif added_edges is None:
                # Assuming that edges were added even if add_edge_list returns None
                print(f"Added {len(edge_list)} edges to the graph.")
            else:
                # Handle other possible return types if any
                print(f"Added edges to the graph. Return type of add_edge_list: {type(added_edges)}")
        except Exception as e:
            print(f"Error during add_edge_list: {e}")
            return

        # Assign edge properties
        if edge_property_cols:
            print("Assigning edge properties in bulk...")
            for prop in edge_property_cols:
                if prop in df_exploded.columns:
                    # Ensure the property exists
                    if prop not in self.graph.edge_properties:
                        # Determine the property type based on data
                        sample_value = df_exploded[prop].dropna().iloc[0]
                        if isinstance(sample_value, str):
                            prop_type = "string"
                        elif isinstance(sample_value, float):
                            prop_type = "float"
                        elif isinstance(sample_value, int):
                            prop_type = "int"
                        else:
                            prop_type = "float"  # Default type
                        self.graph.edge_properties[prop] = self.graph.new_edge_property(prop_type)
                    # Assign the property values
                    self.graph.edge_properties[prop].a = df_exploded[prop].values
            print(f"Assigned edge properties: {edge_property_cols}")

    def print_node_properties(self, node_id, layer):
        """
        Print the properties of a specific node.

        :param node_id: The node ID.
        :param layer: The layer of the node.
        """
        key = (layer, node_id)
        if key in self.node_map:
            v = self.node_map[key]
            properties = {}
            for prop in self.graph.vertex_properties:
                properties[prop] = self.graph.vertex_properties[prop][v]
            print(f"Properties for node {key}:")
            for prop, value in properties.items():
                print(f"  {prop}: {value}")
        else:
            print(f"Node {key} not found in the graph.")

    def visualize_subgraph(self, target_node_id, target_layer, radius=2, filename='subgraph_visualization.png'):
        """
        Visualize a subgraph around the target node.

        :param target_node_id: The node ID around which to visualize the subgraph.
        :param target_layer: The layer of the target node.
        :param radius: The radius (number of hops) around the target node.
        :param filename: Filename to save the visualization image.
        """
        key = (target_layer, target_node_id)
        if key in self.node_map:
            target_vertex = self.node_map[key]
            # Get vertices within the specified radius
            sub_vertices = list(self.graph.get_vertices(radius, [target_vertex]))
            subgraph = self.graph.subgraph(sub_vertices)

            # Define vertex colors based on layer
            layers = subgraph.vertex_properties["layer"]
            unique_layers = list(set(layers.a))
            cmap = plt.get_cmap('tab10')
            layer_to_color = {layer: cmap(i % 10) for i, layer in enumerate(unique_layers)}
            vertex_colors = [layer_to_color.get(layers[v], (0.5, 0.5, 0.5, 1)) for v in subgraph.vertices()]

            # Draw the subgraph
            graph_draw(
                subgraph,
                vertex_fill_color=vertex_colors,
                vertex_text=subgraph.vertex_properties["node_id"],
                output_size=(800, 800),
                vertex_font_size=10,
                edge_pen_width=1.2,
                output=filename
            )
            print(f"Subgraph visualization saved as '{filename}'.")
        else:
            print(f"Node '{target_node_id}' in layer '{target_layer}' not found in the graph.")

    def validate_edge_properties_length(self, edge_properties, num_edges):
        """
        Validates that all lists in edge_properties match the number of edges.
        Raises an error if there is a mismatch.

        :param edge_properties: Dictionary of edge properties.
        :param num_edges: Expected number of edges.
        """
        for prop_name, prop_value in edge_properties.items():
            if isinstance(prop_value, list) and len(prop_value) != num_edges:
                raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match "
                                f"the expected number of edges ({num_edges}).")
            

import pandas as pd
import numpy as np
from graph_tool.all import Graph
import math

class MultilayerNetworkFast:
    def __init__(self, directed=True):
        """
        Initialize the MultilayerNetwork.

        :param directed: Whether the graph is directed. Default is True.
        """
        self.graph = Graph(directed=directed)
        self.node_map = {}  # Maps (layer, node_id) to vertex index

        # Essential vertex properties
        self.graph.vp['node_id'] = self.graph.new_vertex_property('string')
        self.graph.vp['layer'] = self.graph.new_vertex_property('string')

        # Dictionaries to keep track of property types
        self.vertex_properties = {'node_id': 'string', 'layer': 'string'}
        self.edge_properties = {}

    def _infer_property_type(self, series):
        """
        Infer the property type based on a pandas Series.

        :param series: pandas Series.
        :return: String representing the property type ('string', 'int', 'float').
        """
        if pd.api.types.is_float_dtype(series):
            return 'float'
        elif pd.api.types.is_integer_dtype(series):
            return 'int'
        else:
            return 'string'
        
    def _create_vertex_properties(self, properties):
        """
        Create vertex properties dynamically.

        :param properties: Dict of property_name: property_type.
        """
        for prop_name, prop_type in properties.items():
            if prop_name not in self.graph.vp:
                self.graph.vp[prop_name] = self.graph.new_vertex_property(prop_type)
                self.vertex_properties[prop_name] = prop_type

    def _create_edge_properties(self, properties):
        """
        Create edge properties dynamically.

        :param properties: Dict of property_name: property_type.
        """
        for prop_name, prop_type in properties.items():
            if prop_name not in self.graph.ep:
                self.graph.ep[prop_name] = self.graph.new_edge_property(prop_type)
                self.edge_properties[prop_name] = prop_type

    def _expand_edges(self, df, from_node_id_col, to_node_id_col, edge_property_cols, from_layer_col, to_layer_col, split_char):
        records = []
        for _, row in df.iterrows():
            from_nodes = str(row[from_node_id_col]).split(split_char)
            to_nodes = str(row[to_node_id_col]).split(split_char)
            # Ensure edge properties are included
            edge_props = {col: row[col] if col in row else None for col in edge_property_cols}
            for from_node, to_node in product(from_nodes, to_nodes):
                record = {
                    'from_node_id': from_node.strip(),
                    'to_node_id': to_node.strip(),
                    'from_layer': row[from_layer_col],
                    'to_layer': row[to_layer_col]
                }
                record.update(edge_props)
                records.append(record)
        return pd.DataFrame(records)
    
    def add_nodes_from_dataframe(self, df, node_id_col, layer_col, node_property_cols=None, split_char='|'):
        """
        Add nodes to the graph from a DataFrame.
        """
        # Handle splitting
        df[node_id_col] = df[node_id_col].astype(str).str.split(split_char)
        
        # Explode node IDs
        df = df.explode(node_id_col).reset_index(drop=True)
        
        # Remove leading/trailing whitespace from node IDs
        df[node_id_col] = df[node_id_col].str.strip()
        
        # Remove duplicates
        node_cols = [node_id_col, layer_col] + (node_property_cols or [])
        nodes_df = df[node_cols].drop_duplicates().rename(columns={node_id_col: 'node_id', layer_col: 'layer'})
        
        # Infer property types
        property_types = {}
        for col in nodes_df.columns:
            if col not in ['node_id', 'layer']:
                prop_type = self._infer_property_type(nodes_df[col])
                property_types[col] = prop_type
        
        # Create vertex properties
        self._create_vertex_properties(property_types)
        
        # Map nodes to vertex indices
        for idx, row in nodes_df.iterrows():
            key = (row['layer'], row['node_id'])
            if key not in self.node_map:
                v = self.graph.add_vertex()
                self.node_map[key] = v
        
                # Assign essential properties
                self.graph.vp['node_id'][v] = row['node_id']
                self.graph.vp['layer'][v] = row['layer']
        
                # Assign additional properties
                for prop in property_types:
                    self.graph.vp[prop][v] = row[prop]

    def add_edges_from_dataframe(self, df, from_node_id_col, to_node_id_col, from_layer_col, to_layer_col,
                             edge_property_cols=None, split_char='|'):
        # Verify that edge_property_cols exist in df
        if edge_property_cols:
            missing_cols = [col for col in edge_property_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"The following edge_property_cols are missing in the DataFrame: {missing_cols}")
        
        # Generate expanded edges DataFrame
        edges_df = self._expand_edges(df, from_node_id_col, to_node_id_col, edge_property_cols or [], from_layer_col, to_layer_col, split_char)
        
        # Map node keys to vertex indices
        edges_df['from_key'] = list(zip(edges_df['from_layer'], edges_df['from_node_id']))
        edges_df['to_key'] = list(zip(edges_df['to_layer'], edges_df['to_node_id']))
        
        # Filter out edges where nodes are missing in node_map
        edges_df['from_vertex'] = edges_df['from_key'].map(self.node_map)
        edges_df['to_vertex'] = edges_df['to_key'].map(self.node_map)
        edges_df.dropna(subset=['from_vertex', 'to_vertex'], inplace=True)
        
        # Convert vertices to integers
        edges_df['from_vertex'] = edges_df['from_vertex'].astype(int)
        edges_df['to_vertex'] = edges_df['to_vertex'].astype(int)
        
        # Initialize edge properties
        property_types = {}
        if edge_property_cols:
            for col in edge_property_cols:
                if col in edges_df.columns:
                    prop_type = self._infer_property_type(edges_df[col])
                    property_types[col] = prop_type
                else:
                    raise ValueError(f"Edge property column '{col}' not found in the expanded edges DataFrame.")
            # Create edge properties
            self._create_edge_properties(property_types)
        
        # Add edges one by one and assign properties
        for idx, row in edges_df.iterrows():
            from_vertex = int(row['from_vertex'])
            to_vertex = int(row['to_vertex'])
            e = self.graph.add_edge(from_vertex, to_vertex)
            # Assign edge properties
            if edge_property_cols:
                for prop in edge_property_cols:
                    value = row[prop]
                    self.graph.ep[prop][e] = value

    def build_network_from_dataframe(
        self,
        df,
        from_node_id_col,
        to_node_id_col,
        from_layer_col=None,
        to_layer_col=None,
        from_node_property_cols=None,
        to_node_property_cols=None,
        edge_property_cols=None,
        node_split_char='|',
        verbose=True
    ):
        """
        Build the network from a DataFrame with user-specified column mappings.

        :param df: Input DataFrame.
        :param from_node_id_col: Column name for source node IDs.
        :param to_node_id_col: Column name for target node IDs.
        :param from_layer_col: Column name for source node layers (optional).
        :param to_layer_col: Column name for target node layers (optional).
        :param from_node_property_cols: List of column names for source node properties (optional).
        :param to_node_property_cols: List of column names for target node properties (optional).
        :param edge_property_cols: List of column names for edge properties (optional).
        :param node_split_char: Delimiter for splitting multiple node IDs (default: '|').
        :param verbose: Whether to print progress messages (default: True).
        """
        if verbose:
            print("Building network from DataFrame...")

        # Handle missing layer columns by setting default layers
        if from_layer_col is None:
            df['from_layer'] = 'Layer1'
            from_layer_col = 'from_layer'
        if to_layer_col is None:
            df['to_layer'] = 'Layer1'
            to_layer_col = 'to_layer'

        # Extract source nodes
        from_node_cols = [from_node_id_col, from_layer_col] + (from_node_property_cols or [])
        from_nodes = df[from_node_cols].copy()
        from_nodes = from_nodes.rename(columns={from_node_id_col: 'node_id', from_layer_col: 'layer'})

        # Extract target nodes
        to_node_cols = [to_node_id_col, to_layer_col] + (to_node_property_cols or [])
        to_nodes = df[to_node_cols].copy()
        to_nodes = to_nodes.rename(columns={to_node_id_col: 'node_id', to_layer_col: 'layer'})

        # Combine and remove duplicates
        nodes_df = pd.concat([from_nodes, to_nodes], ignore_index=True)
        nodes_df = nodes_df.drop_duplicates()

        # Add nodes to the graph
        if verbose:
            print("Adding nodes to the graph...")
        self.add_nodes_from_dataframe(
            nodes_df,
            node_id_col='node_id',
            layer_col='layer',
            node_property_cols=(from_node_property_cols or []) + (to_node_property_cols or []),
            split_char=node_split_char
        )

        # Add edges to the graph
        if verbose:
            print("Adding edges to the graph...")
        self.add_edges_from_dataframe(
            df,
            from_node_id_col=from_node_id_col,
            to_node_id_col=to_node_id_col,
            from_layer_col=from_layer_col,
            to_layer_col=to_layer_col,
            edge_property_cols=edge_property_cols,
            split_char=node_split_char
        )

        if verbose:
            print("Network construction complete.")


# import pandas as pd
# import numpy as np
# from graph_tool.all import Graph

class MultilayerNetworkSuperFast:
    def __init__(self, directed=True):
        self.graph = Graph(directed=directed)
        self.id_to_index = {}  # Map from custom ID (layer:node_id) to vertex index
        self.index_to_id = {}  # Map from vertex index to custom ID (layer:node_id)
        # Initialize vertex properties
        self.graph.vp['id'] = self.graph.new_vertex_property('string')
        self.graph.vp['layer'] = self.graph.new_vertex_property('string')
        self.graph.vp['node_id'] = self.graph.new_vertex_property('string')

    def _infer_property_type(self, value):
        """
        Infer the property type based on the value.
        """
        if isinstance(value, float):
            return 'float'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, str):
            return 'string'
        else:
            return 'string'  # Fallback to string representation

    def add_vertices_from_dataframe(self, df_nodes, id_col, layer_col, property_cols=None):
        """
        Add vertices from a DataFrame with custom IDs and properties.
        """
        # Create IDs as concatenation of layer and node_id
        ids = df_nodes[layer_col].astype(str) + ':' + df_nodes[id_col].astype(str)
        n_new_vertices = len(ids)
        starting_index = self.graph.num_vertices()
        self.graph.add_vertex(n_new_vertices)

        indices = np.arange(starting_index, starting_index + n_new_vertices, dtype=np.int64)
        # Map IDs to indices
        self.id_to_index.update(zip(ids, indices))
        self.index_to_id.update(zip(indices, ids))

        # Assign IDs to the 'id' vertex property
        id_prop = self.graph.vp['id']
        id_array = np.empty(self.graph.num_vertices(), dtype='U50')  # Adjust the dtype as needed
        if starting_index > 0:
            id_array[:starting_index] = id_prop.a  # Existing IDs
        id_array[starting_index:] = ids.values    # New IDs
        id_prop.a = id_array

        # Assign layer property
        layer_prop = self.graph.vp['layer']
        layer_array = np.empty(self.graph.num_vertices(), dtype='U50')
        if starting_index > 0:
            layer_array[:starting_index] = layer_prop.a
        layer_array[starting_index:] = df_nodes[layer_col].astype(str).values
        layer_prop.a = layer_array

        # Assign node_id property
        node_id_prop = self.graph.vp['node_id']
        node_id_array = np.empty(self.graph.num_vertices(), dtype='U50')
        if starting_index > 0:
            node_id_array[:starting_index] = node_id_prop.a
        node_id_array[starting_index:] = df_nodes[id_col].astype(str).values
        node_id_prop.a = node_id_array

        # Assign additional properties in bulk
        if property_cols:
            for prop_name in property_cols:
                prop_values = df_nodes[prop_name].values
                sample_value = prop_values[0]
                prop_type = self._infer_property_type(sample_value)
                if prop_name not in self.graph.vp:
                    prop = self.graph.new_vertex_property(prop_type)
                    self.graph.vp[prop_name] = prop
                else:
                    prop = self.graph.vp[prop_name]
                # Extend the property array
                prop_array = np.empty(self.graph.num_vertices(), dtype=prop_values.dtype)
                if starting_index > 0:
                    prop_array[:starting_index] = prop.a
                prop_array[starting_index:] = prop_values
                prop.a = prop_array

    #### **2. Updated `add_edges_from_dataframe` Method**

    def add_edges_from_dataframe(self, df_edges, source_id_col, source_layer_col, target_id_col, target_layer_col, property_cols=None):
        """
        Add edges from a DataFrame with custom IDs and properties.
        """
        # Create source and target IDs as concatenated strings
        source_ids = df_edges[source_layer_col].astype(str) + ':' + df_edges[source_id_col].astype(str)
        target_ids = df_edges[target_layer_col].astype(str) + ':' + df_edges[target_id_col].astype(str)

        source_indices = [self.id_to_index.get(id) for id in source_ids]
        target_indices = [self.id_to_index.get(id) for id in target_ids]

        # Filter out edges where source or target is missing
        valid_indices = [i for i, (s, t) in enumerate(zip(source_indices, target_indices)) if s is not None and t is not None]
        if not valid_indices:
            return  # No valid edges to add

        edge_array = np.column_stack((
            [source_indices[i] for i in valid_indices],
            [target_indices[i] for i in valid_indices]
        ))

        if property_cols:
            # Prepare property maps and values
            eprops = []
            prop_values_list = []
            for prop_name in property_cols:
                prop_values = df_edges.iloc[valid_indices][prop_name].values
                sample_value = prop_values[0]
                prop_type = self._infer_property_type(sample_value)
                if prop_name not in self.graph.ep:
                    prop = self.graph.new_edge_property(prop_type)
                    self.graph.ep[prop_name] = prop
                else:
                    prop = self.graph.ep[prop_name]
                eprops.append(prop)
                prop_values_list.append(prop_values)
            # Stack edge array with property values
            edge_list_with_props = np.column_stack((edge_array, *prop_values_list))
            # Add edges with properties
            self.graph.add_edge_list(edge_list_with_props, eprops=eprops)
        else:
            # Add edges without properties
            self.graph.add_edge_list(edge_array)

    def get_vertex_by_custom_id(self, layer, node_id):
        """
        Retrieve a vertex by its custom ID (layer, node_id).
        """
        id_value = layer + ':' + node_id
        v_index = self.id_to_index.get(id_value)
        if v_index is not None:
            return self.graph.vertex(v_index)
        else:
            return None