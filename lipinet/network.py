from graph_tool.all import Graph, GraphView, bfs_search, BFSVisitor, bfs_iterator, shortest_distance, graph_draw
from graph_tool.topology import label_out_component
from collections import deque

import pandas as pd
import numpy as np


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
        self.graph.edge_properties["edge_layertype"] = self.edge_layertype


    def get_set_create_node(self, layer, node_id, create_missing=True, verbose=True, **extra_properties):
        """
        Retrieve an existing node or create a new one if it doesn't exist, then set its properties.

        :param layer: Layer name for the node.
        :param node_id: Unique identifier for the node.
        :param create_missing: If True, create the node if it doesn't exist.
        :param verbose: If True, print debug information.
        :param extra_properties: Additional properties to set on the node.
        :return: The node object if found or created, else None.
        """
        node_key = (layer, node_id)
        node = self.node_map.get(node_key)

        if node is None:
            if create_missing:
                if verbose:
                    print(f"Node {node_key} not found. Creating node.")
                node = self.graph.add_vertex()
                self.node_map[node_key] = node
                self._set_node_property(node, "layer", layer)
                self._set_node_property(node, "node_id", node_id)
            else:
                if verbose:
                    print(f"Node {node_key} not found and create_missing is False.")
                return None

        # Set additional properties
        for prop_name, prop_value in extra_properties.items():
            self._set_node_property(node, prop_name, prop_value)

        if verbose:
            props = { "layer": layer, "node_id": node_id }
            props.update(extra_properties)
            print(f"Node {node_id}: {props}")

        return node


    def add_edge(self, from_node, to_node, from_node_id, to_node_id, from_layer, to_layer, create_missing=True, skip_if_duplicate=None, verbose=True, **edge_properties):
        """
        Add a directed edge between two nodes with specified properties, with optional duplicate handling.

        :param from_node: The source node (vertex) of the edge.
        :param to_node: The target node (vertex) of the edge.
        :param from_node_id: The ID of the source node.
        :param to_node_id: The ID of the target node.
        :param from_layer: The layer of the source node.
        :param to_layer: The layer of the target node.
        :param create_missing: If False, raises an error if either node does not exist.
        :param skip_if_duplicate: If None, allows duplicate edges; if 'any', skips if any edge exists;
                                if 'exact', skips only if an edge with identical properties exists.
        :param edge_properties: Additional properties to apply to the edge.
        :param verbose: If True, prints debugging information.
        :return: The created edge object, or None if the edge was skipped.

        Raises:
        - RuntimeError: If `create_missing` is False and a node does not exist in the node map.
        """
        # Check node existence if create_missing is False
        if not create_missing:
            if from_node is None:
                raise RuntimeError(f"Node '{from_node_id}' in layer '{from_layer}' does not exist. "
                                f"Set `create_missing=True` to create missing nodes.")
            if to_node is None:
                raise RuntimeError(f"Node '{to_node_id}' in layer '{to_layer}' does not exist. "
                                f"Set `create_missing=True` to create missing nodes.")

        # Handle duplicate edge behavior based on skip_if_duplicate
        existing_edges = self.graph.edge(from_node, to_node, all_edges=True)
        if existing_edges:
            if skip_if_duplicate == "any":
                # Skip if any edge exists between the nodes
                if verbose:
                    print(f"Edge from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) already exists. Skipping due to 'any' duplicate policy.")
                return None

            elif skip_if_duplicate == "exact":
                # Check if an exact edge with identical properties exists
                for edge in existing_edges:
                    identical = all(
                        (prop in self.graph.edge_properties and self.graph.edge_properties[prop][edge] == value)
                        for prop, value in edge_properties.items()
                    )
                    if identical:
                        if verbose:
                            print(f"Identical edge from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) exists. Skipping due to 'exact' duplicate policy.")
                        return None

        # Create the edge between nodes
        edge = self.graph.add_edge(from_node, to_node)

        # Set edge properties using _set_edge_property
        for prop_name, prop_value in edge_properties.items():
            self._set_edge_property(edge, prop_name, prop_value)

        if verbose:
            print(f"Edge added from {from_node_id} ({from_layer}) to {to_node_id} ({to_layer}) with properties: {edge_properties}")

        return edge

    
    def add_edges_from_pairs(self, node_pairs, split_char='|', create_missing=False, skip_if_duplicate='exact', **edge_properties):
        """
        Add directed edges between pairs of nodes using node_pairs, with optional per-edge properties.

        Args:
        - node_pairs: A list of tuples (node1_id, node2_id), where each node is identified by (layer, node_id).
        - split_char: Character to split multiple node IDs.
        - create_missing: If True, creates missing nodes if they are not found in node_map.
        - edge_properties: Additional properties for each created edge. Properties can be single values or lists.
        """
        # Calculate expected number of edges
        expected_num_edges = sum(
            len([id_.strip() for id_ in node1_id[1].split(split_char)]) *
            len([id_.strip() for id_ in node2_id[1].split(split_char)])
            for node1_id, node2_id in node_pairs if node1_id and node2_id
        )

        # Validate list lengths
        self.validate_edge_properties_length(edge_properties, expected_num_edges)

        edge_count = 0  # Counter for each edge added, used as index for property lists

        for node1_id, node2_id in node_pairs:
            if node1_id is None or node2_id is None:
                continue
            
            node1_ids = [id_.strip() for id_ in node1_id[1].split(split_char)] if split_char in node1_id[1] else [node1_id[1]]
            node2_ids = [id_.strip() for id_ in node2_id[1].split(split_char)] if split_char in node2_id[1] else [node2_id[1]]

            for id1 in node1_ids:
                for id2 in node2_ids:
                    from_layer = node1_id[0]
                    to_layer = node2_id[0]
                    node1 = self.get_set_create_node(layer=from_layer, node_id=id1, create_missing=create_missing)
                    node2 = self.get_set_create_node(layer=to_layer, node_id=id2, create_missing=create_missing)

                    if node1 is None or node2 is None:
                        continue
                    
                    # Resolve edge-specific properties using helper function
                    edge_specific_properties = self.resolve_edge_properties(edge_properties, edge_count)
                    
                    # Use add_edge to create a single edge with resolved properties
                    self.add_edge(
                        node1, node2,
                        from_node_id=node1_ids, to_node_id=node2_ids,
                        from_layer=from_layer, to_layer=to_layer,
                        create_missing=create_missing,
                        skip_if_duplicate=skip_if_duplicate,
                        **edge_specific_properties
                    )
                    edge_count += 1


    def add_edges_from_nodes(
        self,
        from_nodes,
        to_nodes,
        from_layer,
        to_layer,
        split_char='|',
        create_missing=False,
        skip_if_duplicate='exact',
        verbose=True,
        **edge_properties
        ):
            """
            Adds directed edges from 'from_nodes' to 'to_nodes' using specified layers and optional per-edge properties.
            Validates that any list properties match the expected number of edges.

            Args:
            - from_nodes: List or iterable of node IDs for 'from' nodes.
            - to_nodes: List or iterable of node IDs for 'to' nodes.
            - from_layer: Layer name for all 'from' nodes.
            - to_layer: Layer name for all 'to' nodes.
            - split_char: Character to split multiple node IDs.
            - create_missing: If True, creates missing nodes if they are not found in node_map.
            - skip_if_duplicate: If 'exact', skips adding duplicate edges based on exact matches. Other options can be implemented as needed.
            - verbose: If True, prints debug information.
            - edge_properties: Additional properties for each created edge. Properties can be single values or lists.

            Raises:
            - ValueError: If any list property in edge_properties does not match the expected number of edges.
            """
            if len(from_nodes) != len(to_nodes):
                raise ValueError("from_nodes and to_nodes must have the same length.")

            # Initialize counters
            expected_num_edges = 0
            skipped_rows_initial = 0
            skipped_rows = []

            # First pass: Calculate the expected number of edges, safely handling non-string node IDs
            for from_node_id, to_node_id in zip(from_nodes, to_nodes):
                if isinstance(from_node_id, str) and isinstance(to_node_id, str) and from_node_id and to_node_id:
                    from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)]
                    to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)]
                    expected_num_edges += len(from_node_ids) * len(to_node_ids)
                else:
                    skipped_rows_initial += 1
                    skipped_rows.append(f"Skipped: From node: {from_node_id}, To node: {from_node_id}")

            if verbose and skipped_rows_initial > 0:
                print(f"Skipped {skipped_rows_initial} rows due to non-string node IDs or missing values.\n{skipped_rows}")

            # Validate list lengths in edge_properties
            self.validate_edge_properties_length(edge_properties, expected_num_edges)

            edge_count = 0  # Counter for each edge added
            skipped_rows = 0  # Counter for skipped rows during edge addition

            # Second pass: Iterate through from_nodes and to_nodes to add edges
            for idx, (from_node_id, to_node_id) in enumerate(zip(from_nodes, to_nodes)):
                # Check if both node IDs are strings
                if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
                    if verbose:
                        print(f"Skipping row {idx} due to non-string node IDs: from_node_id={from_node_id}, to_node_id={to_node_id}")
                    skipped_rows += 1
                    continue

                # Split node IDs by split_char and strip whitespace
                from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
                to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

                # Iterate through all combinations of from_node_ids and to_node_ids
                for id1 in from_node_ids:
                    for id2 in to_node_ids:
                        # Retrieve or create nodes
                        node1 = self.get_set_create_node(
                            layer=from_layer,
                            node_id=id1,
                            create_missing=create_missing,
                            verbose=verbose
                            # **edge_properties
                        )
                        node2 = self.get_set_create_node(
                            layer=to_layer,
                            node_id=id2,
                            create_missing=create_missing,
                            verbose=verbose
                            # **edge_properties
                        )

                        # Skip if either node couldn't be retrieved or created
                        if node1 is None or node2 is None:
                            if verbose:
                                print(f"Skipping edge from '{id1}' to '{id2}' because one of the nodes could not be retrieved or created.")
                            continue

                        # Resolve edge-specific properties using helper function
                        edge_specific_properties = self.resolve_edge_properties(edge_properties, edge_count)

                        # Add the edge with resolved properties
                        self.add_edge(
                            node1, node2,
                            from_node_id=from_node_id,
                            to_node_id=to_node_id,
                            from_layer=from_layer,
                            to_layer=to_layer,
                            create_missing=create_missing,
                            skip_if_duplicate=skip_if_duplicate,
                            verbose=verbose,
                            **edge_specific_properties
                        )

                        if verbose:
                            print(f"Added edge from '{id1}' to '{id2}' with properties {edge_specific_properties}")

                        edge_count += 1

            if verbose:
                print(f"Total edges expected to add: {expected_num_edges}")
                print(f"Total edges actually added: {edge_count}")
                if skipped_rows > 0:
                    print(f"Total rows skipped during edge addition: {skipped_rows}")


    def add_edges_from_dataframe(
        self,
        df: pd.DataFrame,
        from_col: str,
        to_col: str,
        from_layer: str = None,
        to_layer: str = None,
        from_layer_col: str = None,
        to_layer_col: str = None,
        edge_property_cols: list = None,
        from_node_property_cols: list = None,
        to_node_property_cols: list = None,
        split_char: str = '|',
        create_missing: bool = False,
        skip_if_duplicate: str = 'exact',
        verbose: bool = True,
        edge_property_mode: str = 'per-edge'  # New parameter
    ):
        """
        Adds directed edges from a DataFrame, using specified columns for node IDs and optional layers.
        Allows customizable edge properties by specifying which DataFrame columns to use.
        Additionally, allows setting or updating custom properties of source and target nodes
        based on specified DataFrame columns.
        Supports multiple IDs in a single cell, separated by a specified character.

        Args:
        - df (pd.DataFrame): DataFrame containing edge and node information.
        - from_col (str): Column name in df for source node IDs.
        - to_col (str): Column name in df for target node IDs.
        - from_layer (str, optional): Fixed layer name for all source nodes if `from_layer_col` is not specified.
        - to_layer (str, optional): Fixed layer name for all target nodes if `to_layer_col` is not specified.
        - from_layer_col (str, optional): Column in df specifying 'from' node layers if layers vary by row.
        - to_layer_col (str, optional): Column in df specifying 'to' node layers if layers vary by row.
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

        Raises:
        - ValueError: If no fixed layer name or column name is provided for either source or target layers.
        - ValueError: If edge_property_mode is not one of the accepted values.
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

        # Prepare node property columns
        from_node_properties = from_node_property_cols if from_node_property_cols else []
        to_node_properties = to_node_property_cols if to_node_property_cols else []

        # Ensure that either a fixed layer or a column is provided for source and target layers
        if from_layer is None and from_layer_col is None:
            raise ValueError("Either `from_layer` or `from_layer_col` must be specified for source node layers.")
        if to_layer is None and to_layer_col is None:
            raise ValueError("Either `to_layer` or `to_layer_col` must be specified for target node layers.")

        # Calculate the expected number of edges for property validation
        expected_num_edges = 0
        skipped_due_to_invalid = 0

        for idx, row in df.iterrows():
            from_node_id = row[from_col]
            to_node_id = row[to_col]

            if pd.isna(from_node_id) or pd.isna(to_node_id):
                skipped_due_to_invalid += 1
                continue

            if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
                skipped_due_to_invalid += 1
                continue

            from_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
            to_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

            expected_num_edges += len(from_ids) * len(to_ids)

        # Validate list lengths in edge_properties based on mode
        if edge_property_mode == 'per-edge':
            # Each property list must match the number of edges
            self.validate_edge_properties_length(edge_properties, expected_num_edges)
        elif edge_property_mode == 'per-row':
            # Each property list must match the number of rows (excluding skipped)
            num_valid_rows = len(df) - skipped_due_to_invalid
            for prop_name, prop_value in edge_properties.items():
                if isinstance(prop_value, list):
                    if len(prop_value) != len(df):
                        raise ValueError(f"Length of property list '{prop_name}' ({len(prop_value)}) does not match the number of rows ({len(df)}).")
        # Note: 'per-row' mode assumes that the property value for a row applies to all edges generated from that row.

        # Initialize counters and trackers
        edge_count = 0  # Counter for each edge added, used as index for property lists
        total_edges_added = 0

        # Iterate through each row to add edges
        for idx, row in df.iterrows():
            from_node_id = row[from_col]
            to_node_id = row[to_col]

            if pd.isna(from_node_id) or pd.isna(to_node_id):
                if verbose:
                    if pd.isna(from_node_id) and pd.isna(to_node_id):
                        print(f"Skipping row {idx}: Both 'from_col' and 'to_col' are NaN. Row data: {row.to_dict()}")
                    elif pd.isna(from_node_id):
                        print(f"Skipping row {idx}: 'from_col' is NaN. Row data: {row.to_dict()}")
                    elif pd.isna(to_node_id):
                        print(f"Skipping row {idx}: 'to_col' is NaN. Row data: {row.to_dict()}")
                skipped_due_to_invalid += 1
                continue

            if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
                if verbose:
                    issues = []
                    if not isinstance(from_node_id, str):
                        issues.append(f"'from_col' is of type {type(from_node_id).__name__} with value {from_node_id}")
                    if not isinstance(to_node_id, str):
                        issues.append(f"'to_col' is of type {type(to_node_id).__name__} with value {to_node_id}")
                    issue_details = "; ".join(issues)
                    print(f"Skipping row {idx}: {issue_details}. Row data: {row.to_dict()}")
                skipped_due_to_invalid += 1
                continue

            # Split node IDs if necessary
            from_node_ids = [id_.strip() for id_ in from_node_id.split(split_char)] if split_char in from_node_id else [from_node_id]
            to_node_ids = [id_.strip() for id_ in to_node_id.split(split_char)] if split_char in to_node_id else [to_node_id]

            # Determine effective layers
            effective_from_layer = from_layer if from_layer else row.get(from_layer_col, from_layer)
            effective_to_layer = to_layer if to_layer else row.get(to_layer_col, to_layer)

            # Extract node properties from the row
            from_node_props = {prop: row[prop] for prop in from_node_properties if prop in row and not pd.isna(row[prop])}
            to_node_props = {prop: row[prop] for prop in to_node_properties if prop in row and not pd.isna(row[prop])}

            # Resolve edge-specific properties based on mode
            if edge_property_mode == 'per-edge':
                # For per-edge mode, properties are indexed by edge_count
                edge_specific_properties = {}
                for prop in edge_properties:
                    prop_value = edge_properties[prop]
                    if isinstance(prop_value, list):
                        edge_specific_properties[prop] = prop_value[edge_count]
                    else:
                        edge_specific_properties[prop] = prop_value
            elif edge_property_mode == 'per-row':
                # For per-row mode, properties are indexed by row index
                edge_specific_properties = {}
                for prop in edge_properties:
                    prop_value = edge_properties[prop]
                    if isinstance(prop_value, list):
                        edge_specific_properties[prop] = prop_value[idx]
                    else:
                        edge_specific_properties[prop] = prop_value

            # Iterate through all combinations of from_node_ids and to_node_ids
            for id1 in from_node_ids:
                for id2 in to_node_ids:
                    # Retrieve or create source node with custom properties
                    node1 = self.get_set_create_node(
                        layer=effective_from_layer,
                        node_id=id1,
                        create_missing=create_missing,
                        verbose=verbose,
                        **from_node_props
                    )

                    # Retrieve or create target node with custom properties
                    node2 = self.get_set_create_node(
                        layer=effective_to_layer,
                        node_id=id2,
                        create_missing=create_missing,
                        verbose=verbose,
                        **to_node_props
                    )

                    if node1 is None or node2 is None:
                        if verbose:
                            print(f"Skipping edge from '{id1}' to '{id2}': Missing nodes.")
                        continue

                    # Add the edge with resolved properties
                    self.add_edge(
                        from_node=node1,
                        to_node=node2,
                        from_node_id=id1,
                        to_node_id=id2,
                        from_layer=effective_from_layer,
                        to_layer=effective_to_layer,
                        create_missing=create_missing,
                        skip_if_duplicate=skip_if_duplicate,
                        **edge_specific_properties
                    )
                    edge_count += 1
                    total_edges_added += 1

        # Summary of the operation - doesn't seem to be working as expected - underestimates expected number of edges to be added, and overestimates number skipped rows
        # if verbose:
        #     print(f"Total edges expected to be added: {expected_num_edges}")
        #     print(f"Total edges actually added: {total_edges_added}")
        #     if skipped_due_to_invalid > 0:
        #         print(f"Total rows skipped due to invalid node IDs: {skipped_due_to_invalid}")


    def resolve_edge_properties(self, edge_properties, edge_index):
        """
        Resolve edge properties for a specific edge, handling both single values and lists.

        :param edge_properties: Dictionary of edge properties, where values can be lists or single values.
        :param edge_index: The index of the current edge being processed.
        :return: A dictionary of properties specific to the current edge.
        """
        resolved_properties = {}
        for prop_name, prop_value in edge_properties.items():
            if isinstance(prop_value, list):
                resolved_properties[prop_name] = prop_value[edge_index]
            else:
                resolved_properties[prop_name] = prop_value
        return resolved_properties


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


    def _set_edge_property(self, edge, prop_name, value, nan_replacement=None):
        """
        Set or create an edge property for a given edge.
        
        :param edge: The edge to assign the property.
        :param prop_name: The name of the property.
        :param value: The value of the property, which determines the property type.
        :param nan_replacement: Optional; the value to replace NaN with (default is None).
        """
        import math
        # Replace NaN with the specified replacement, if value is NaN
        if isinstance(value, float) and math.isnan(value):
            value = nan_replacement

        # Check if the property map exists, create it if not
        if prop_name not in self.graph.ep:
            # Determine type based on the value type
            if isinstance(value, str):
                prop_type = "string"
            elif isinstance(value, float):
                prop_type = "float"
            elif isinstance(value, int):
                prop_type = "int"
            else:
                prop_type = "object"  # For more complex data types

            self.graph.ep[prop_name] = self.graph.new_edge_property(prop_type)
        
        # Set the property value for the edge
        self.graph.ep[prop_name][edge] = value


    def build_layer(self, nodes, layer_name, custom_node_properties={}, verbose=True):
        """
        Build a network layer from a dataset.
        
        :param layer_data: List of desired node ids
        :param layer_name: Name of the layer
        :param custom_properties: Dictionary of extra properties for each node {node_id: {property_name: value}}
        :param verbose: Whether to print every node update.
        """
        custom_node_properties = custom_node_properties or {}
        for node_id in nodes:
            # Get extra properties for this node if they exist
            individual_node_properties = custom_node_properties.get(node_id, {})
            self.get_set_create_node(layer=layer_name, node_id=node_id, verbose=verbose, create_missing=True, **individual_node_properties)
    
    
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
