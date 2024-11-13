# Author: Macabe Daley

from graph_tool.all import Graph, GraphView, bfs_search, BFSVisitor, bfs_iterator, shortest_distance, graph_draw
from graph_tool.topology import label_out_component
from collections import deque

import pandas as pd
import numpy as np
# import json
# import math
# import pickle

from itertools import product

from typing import Any, Dict, Tuple, List

# Import for the `display` function used in the `grow_onion` method
try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


# A super fast alternative to the previous implementation. 
# Less flexible, but capable of creating networks from huge datasets (e.g. 10m+ nodes) 
# in only minutes. So long as you have the memory to hold the dfs and they are relatively clean.
# Relies on the use of mapping numericals and categoricals, plus some graph-tool tricks.

# Future dev:
# - polar integration


class SuperOnion:
    def __init__(self, directed=True):
        self.graph = Graph(directed=directed)
        self.custom_id_to_vertex_index: Dict[Tuple[int, int], int] = {}    # Map from custom ID tuple (layer_code, node_id_int) to vertex index
        self.vertex_index_to_custom_id: Dict[int, Tuple[int, int]] = {}    # Map from vertex index to custom ID tuple (layer_code, node_id_int)
        self.layer_code_to_name: Dict[int, str] = {}           # Map from layer codes to layer names
        self.node_id_int_to_str: Dict[int, str] = {}           # Map from node_id_int to node_id strings
        self.layer_name_to_code: Dict[str, int] = {}           # Map from layer names to integer codes
        self.node_id_str_to_int: Dict[str, int] = {}           # Map from node_id strings to integer codes
        
        # Initialize vertex properties for layer and node ID hashes
        self.graph.vp['layer_hash'] = self.graph.new_vertex_property('int64_t')
        self.graph.vp['node_id_hash'] = self.graph.new_vertex_property('int64_t')
        
        # Initialize dictionaries to store categorical property mappings
        self.vertex_categorical_mappings = {}  # {prop_name: {'str_to_int': {}, 'int_to_str': {}}}
        self.edge_categorical_mappings = {}    # {prop_name: {'str_to_int': {}, 'int_to_str': {}}}
    
    def _infer_property_type(self, value):
        """
        Infer the property type based on the value.
        """
        if isinstance(value, (int, np.integer)):
            return 'int'
        elif isinstance(value, (float, np.floating)):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, (bool, np.bool)):
            return 'bool'
        else:
            return 'object'  # Use 'object' for any other type
    
    def _map_categorical_property(self, prop_name, values, category_type='vertex'):
        """
        Map categorical string values to unique integer codes.
        
        Parameters:
            prop_name (str): Name of the property.
            values (array-like): Array of string values.
            category_type (str): 'vertex' or 'edge' to determine the mapping dictionary.
        
        Returns:
            np.ndarray: Array of integer codes corresponding to the categorical values.
        """
        if category_type == 'vertex':
            mappings = self.vertex_categorical_mappings.get(prop_name, {'str_to_int': {}, 'int_to_str': {}})
        else:
            mappings = self.edge_categorical_mappings.get(prop_name, {'str_to_int': {}, 'int_to_str': {}})
        
        # Initialize a list to store mapped integer codes
        mapped_values = np.empty(len(values), dtype=np.int32)
        
        # Iterate through the values and assign unique integer codes
        current_code = len(mappings['str_to_int'])
        for i, val in enumerate(values):
            if val in mappings['str_to_int']:
                mapped_values[i] = mappings['str_to_int'][val]
            else:
                mappings['str_to_int'][val] = current_code
                mappings['int_to_str'][current_code] = val
                mapped_values[i] = current_code
                current_code += 1
        
        # Update the mapping dictionaries
        if category_type == 'vertex':
            self.vertex_categorical_mappings[prop_name] = mappings
        else:
            self.edge_categorical_mappings[prop_name] = mappings
        
        return mapped_values
    
    def _map_layer(self, layer_name):
        """
        Map a layer name to an integer code, updating the mapping if necessary.
        """
        if layer_name in self.layer_name_to_code:
            return self.layer_name_to_code[layer_name]
        else:
            # Assign a new integer code
            layer_code = len(self.layer_name_to_code)
            self.layer_name_to_code[layer_name] = layer_code
            self.layer_code_to_name[layer_code] = layer_name
            return layer_code
    
    def _map_node_id(self, node_id_str):
        """
        Map a node ID string to an integer code, updating the mapping if necessary.
        """
        if node_id_str in self.node_id_str_to_int:
            return self.node_id_str_to_int[node_id_str]
        else:
            # Assign a new integer code
            node_id_int = len(self.node_id_str_to_int)
            self.node_id_str_to_int[node_id_str] = node_id_int
            self.node_id_int_to_str[node_id_int] = node_id_str
            return node_id_int
        
    def grow_onion(
        self, 
        df_nodes: pd.DataFrame, 
        df_edges: pd.DataFrame, 
        node_prop_cols: List[str] = ['node_prop_1', 'node_prop_2'], 
        edge_prop_cols: List[str] = ['edge_prop_1', 'edge_prop_2'],
        drop_na: bool = True,
        drop_duplicates: bool = True,
        use_display: bool = True,
        node_id_col: str = 'node_id',
        node_layer_col: str = 'layer',
        edge_source_id_col: str = 'source_id',
        edge_source_layer_col: str = 'source_layer',
        edge_target_id_col: str = 'target_id',
        edge_target_layer_col: str = 'target_layer'
    ) -> None:
        """
        Grow the onion graph by adding nodes and edges from provided DataFrames.

        This method performs the following steps:
            1. Displays a snippet and shape of the node and edge DataFrames.
            2. Adds vertices from the node DataFrame.
            3. Adds edges from the edge DataFrame.
            4. Displays a summary of the graph.
            5. Lists all vertex and edge properties.

        Parameters:
            df_nodes (pd.DataFrame): DataFrame containing node information.
            df_edges (pd.DataFrame): DataFrame containing edge information.
            node_prop_cols (List[str], optional): List of node property column names to include. Defaults to ['node_prop_1', 'node_prop_2'].
            edge_prop_cols (List[str], optional): List of edge property column names to include. Defaults to ['edge_prop_1', 'edge_prop_2'].
            drop_na (bool, optional): Whether to drop rows with missing IDs or layers. Defaults to True.
            drop_duplicates (bool, optional): Whether to drop nodes and edges that have duplicate entries. Only consideres layer and node ids for vertices and edges, not their properties. Defaults to True.
            use_display (bool, optional): Whether to use the `display` function (useful in Jupyter notebooks). If False, uses `print`. Defaults to True.
            node_id_col (str, optional): Column name for node IDs in df_nodes. Defaults to 'node_id'.
            node_layer_col (str, optional): Column name for layer names in df_nodes. Defaults to 'layer'.
            edge_source_id_col (str, optional): Column name for source node IDs in df_edges. Defaults to 'source_id'.
            edge_source_layer_col (str, optional): Column name for source layer names in df_edges. Defaults to 'source_layer'.
            edge_target_id_col (str, optional): Column name for target node IDs in df_edges. Defaults to 'target_id'.
            edge_target_layer_col (str, optional): Column name for target layer names in df_edges. Defaults to 'target_layer'.

        Raises:
            ValueError: If any of the specified columns are missing from the provided DataFrames.

        Returns:
            None
        """
        # Validate that the necessary columns exist in df_nodes
        required_node_cols = [node_id_col, node_layer_col] + node_prop_cols
        missing_node_cols = [col for col in required_node_cols if col not in df_nodes.columns]
        if missing_node_cols:
            raise ValueError(f"The following node columns are missing in df_nodes: {missing_node_cols}")
        
        # Validate that the necessary columns exist in df_edges
        required_edge_cols = [edge_source_id_col, edge_source_layer_col, edge_target_id_col, edge_target_layer_col] + edge_prop_cols
        missing_edge_cols = [col for col in required_edge_cols if col not in df_edges.columns]
        if missing_edge_cols:
            raise ValueError(f"The following edge columns are missing in df_edges: {missing_edge_cols}")

        if drop_duplicates:
            # Optional: Remove duplicate nodes and edges
            df_nodes = df_nodes.drop_duplicates(subset=[node_id_col, node_layer_col])
            df_edges = df_edges.drop_duplicates(subset=[edge_source_id_col, edge_source_layer_col, edge_target_id_col, edge_target_layer_col])

        # Display snippet of the data and shape
        for df, name in zip([df_nodes, df_edges], ['Nodes', 'Edges']):
            if use_display and IPYTHON_AVAILABLE:
                display(df.head())
            else:
                print(f"{name} DataFrame Head:\n{df.head()}\n")
            print(f"{name} DataFrame Shape: {df.shape}\n")    
    
        # Add vertices from the DataFrame
        self.add_vertices_from_dataframe(
            df_nodes,
            id_col=node_id_col,      # Custom node ID column
            layer_col=node_layer_col,     # Custom layer column
            property_cols=node_prop_cols,
            drop_na=drop_na,           # Drop rows with missing IDs or layers
            string_override=True
        )
    
        # Add edges from the DataFrame
        self.add_edges_from_dataframe(
            df_edges,
            source_id_col=edge_source_id_col,
            source_layer_col=edge_source_layer_col,
            target_id_col=edge_target_id_col,
            target_layer_col=edge_target_layer_col,
            property_cols=edge_prop_cols,
            drop_na=drop_na,
            string_override=True
        )
    
        # Display graph summary
        summary_info = self.summary()
        if use_display and IPYTHON_AVAILABLE:
            print("\nGraph Summary:")
            for key, value in summary_info.items():
                if isinstance(value, list):
                    print(f"{key}:")
                    for item in value:
                        print(f"  - {item}")
                else:
                    print(f"{key}: {value}")
        else:
            print("\nGraph Summary:")
            for key, value in summary_info.items():
                if isinstance(value, list):
                    print(f"{key}:")
                    for item in value:
                        print(f"  - {item}")
                else:
                    print(f"{key}: {value}")
    
        # List all vertex properties
        vertex_props_df = self.list_vertex_properties()
        if use_display and IPYTHON_AVAILABLE:
            print("\nVertex Properties:")
            display(vertex_props_df)
        else:
            print("\nVertex Properties:")
            print(vertex_props_df)
    
        # List all edge properties
        edge_props_df = self.list_edge_properties()
        if use_display and IPYTHON_AVAILABLE:
            print("\nEdge Properties:")
            display(edge_props_df)
        else:
            print("\nEdge Properties:")
            print(edge_props_df)
    
    def add_vertices_from_dataframe(self, df_nodes, id_col, layer_col, property_cols=None, drop_na=True, fill_na_with=None, string_override=False):
        """
        Add vertices from a DataFrame with custom IDs and properties.
        
        Parameters:
            df_nodes (pd.DataFrame): DataFrame containing node information.
            id_col (str): Column name for node IDs.
            layer_col (str): Column name for layer names.
            property_cols (list, optional): List of property column names to include.
            drop_na (bool, optional): Whether to drop rows with missing IDs or layers.
            fill_na_with (any, optional): Value to fill NaNs if not dropping.
        """
        df_nodes = df_nodes.copy()
        # Handle missing values
        if drop_na:
            df_nodes = df_nodes.dropna(subset=[id_col, layer_col])
        else:
            df_nodes = df_nodes.fillna({id_col: fill_na_with, layer_col: fill_na_with})
        
        # Map layers and node IDs to integer codes
        df_nodes['layer_int'] = df_nodes[layer_col].apply(self._map_layer)
        df_nodes['node_id_int'] = df_nodes[id_col].apply(self._map_node_id)
        
        # Create custom ID tuples
        custom_ids = list(zip(df_nodes['layer_int'], df_nodes['node_id_int']))
        n_new_vertices = len(custom_ids)
        starting_index = self.graph.num_vertices()
        self.graph.add_vertex(n_new_vertices)
        
        # Update mapping dictionaries
        new_indices = np.arange(starting_index, starting_index + n_new_vertices, dtype=np.int64)
        self.custom_id_to_vertex_index.update(zip(custom_ids, new_indices))
        self.vertex_index_to_custom_id.update(zip(new_indices, custom_ids))
        
        # Assign 'layer_hash' and 'node_id_hash' properties in bulk
        self.graph.vp['layer_hash'].a[starting_index:] = df_nodes['layer_int'].values
        self.graph.vp['node_id_hash'].a[starting_index:] = df_nodes['node_id_int'].values
        
        # Assign additional properties
        if property_cols:
            for prop_name in property_cols:
                prop_values = df_nodes[prop_name].values
                sample_value = prop_values[0]
                prop_type = self._infer_property_type(sample_value)
                
                if prop_type in ['int', 'float'] and string_override!=True:
                    if prop_name not in self.graph.vp:
                        prop = self.graph.new_vertex_property(prop_type)
                        self.graph.vp[prop_name] = prop
                    else:
                        prop = self.graph.vp[prop_name]
                    
                    # Assign values in bulk
                    prop.a[starting_index:] = prop_values
                elif prop_type in ['string', 'bool'] or string_override==True:
                    # Map categorical string values to integers
                    mapped_values = self._map_categorical_property(prop_name, prop_values, category_type='vertex')
                    
                    if prop_name not in self.graph.vp:
                        prop = self.graph.new_vertex_property('int')  # Store mapped integers as 'int'
                        self.graph.vp[prop_name] = prop
                    else:
                        prop = self.graph.vp[prop_name]
                    
                    # Assign mapped integer codes
                    prop.a[starting_index:] = mapped_values
                else:
                    print(f"Unsupported property type for vertex property '{prop_name}': {prop_type}")
                    pass  # Extend as needed

        # Update layer_code_to_name and node_id_int_to_str mappings
        for layer_name in df_nodes[layer_col].unique():
            layer_code = self.layer_name_to_code.get(layer_name)
            if layer_code is not None:
                self.layer_code_to_name[layer_code] = layer_name

        for node_id_str in df_nodes[id_col].unique():
            node_id_int = self.node_id_str_to_int.get(node_id_str)
            if node_id_int is not None:
                self.node_id_int_to_str[node_id_int] = node_id_str

        # Update the cached node_map if it exists
        if hasattr(self, '_node_map_cache'):
            for (layer_code, node_id_int), vertex_index in zip(custom_ids, new_indices):
                layer_name = self.layer_code_to_name.get(layer_code, f"Unknown Layer ({layer_code})")
                node_id_str = self.node_id_int_to_str.get(node_id_int, f"Unknown ID ({node_id_int})")
                self._node_map_cache[(layer_name, node_id_str)] = vertex_index
    
    def add_edges_from_dataframe(self, df_edges, source_id_col, source_layer_col, target_id_col, target_layer_col, property_cols=None, drop_na=True, fill_na_with=None, string_override=False):
        """
        Add edges from a DataFrame with custom IDs and properties.
        
        Parameters:
            df_edges (pd.DataFrame): DataFrame containing edge information.
            source_id_col (str): Column name for source node IDs.
            source_layer_col (str): Column name for source layer names.
            target_id_col (str): Column name for target node IDs.
            target_layer_col (str): Column name for target layer names.
            property_cols (list, optional): List of property column names to include.
            drop_na (bool, optional): Whether to drop rows with missing IDs or layers.
            fill_na_with (any, optional): Value to fill NaNs if not dropping.
        """
        df_edges = df_edges.copy()
        # Handle missing values
        if drop_na:
            df_edges = df_edges.dropna(subset=[source_id_col, source_layer_col, target_id_col, target_layer_col])
        else:
            df_edges = df_edges.fillna({
                source_id_col: fill_na_with,
                source_layer_col: fill_na_with,
                target_id_col: fill_na_with,
                target_layer_col: fill_na_with
            })
        
        # Map layers and node IDs to integer codes
        df_edges['source_layer_int'] = df_edges[source_layer_col].apply(self._map_layer)
        df_edges['source_id_int'] = df_edges[source_id_col].apply(self._map_node_id)
        df_edges['target_layer_int'] = df_edges[target_layer_col].apply(self._map_layer)
        df_edges['target_id_int'] = df_edges[target_id_col].apply(self._map_node_id)
        
        # Create source and target ID tuples
        source_ids = list(zip(df_edges['source_layer_int'], df_edges['source_id_int']))
        target_ids = list(zip(df_edges['target_layer_int'], df_edges['target_id_int']))
        
        # Map IDs to vertex indices
        source_indices = [self.custom_id_to_vertex_index.get(id_tuple) for id_tuple in source_ids]
        target_indices = [self.custom_id_to_vertex_index.get(id_tuple) for id_tuple in target_ids]
        
        # Filter out edges where source or target is missing
        valid_indices = [i for i, (s, t) in enumerate(zip(source_indices, target_indices)) if s is not None and t is not None]
        if not valid_indices:
            print("No valid edges to add.")
            return
        
        # Prepare edge list
        edge_array = np.column_stack((
            [source_indices[i] for i in valid_indices],
            [target_indices[i] for i in valid_indices]
        ))
        
        # Initialize lists to hold edge properties
        eprops = []
        prop_values_list = []
        
        # Process properties
        if property_cols:
            for prop_name in property_cols:
                prop_values = df_edges.iloc[valid_indices][prop_name].values
                sample_value = prop_values[0]
                prop_type = self._infer_property_type(sample_value)
                
                if prop_type in ['int', 'float'] and string_override!=True:
                    if prop_name not in self.graph.ep:
                        prop = self.graph.new_edge_property(prop_type)
                        self.graph.ep[prop_name] = prop
                    else:
                        prop = self.graph.ep[prop_name]
                    
                    # Collect prop_values
                    prop_values_list.append(prop_values)
                    eprops.append(prop)
                elif prop_type in ['string', 'bool'] or string_override==True:
                    # Map categorical string values to integers
                    mapped_values = self._map_categorical_property(prop_name, prop_values, category_type='edge')
                    
                    if prop_name not in self.graph.ep:
                        prop = self.graph.new_edge_property('int')  # Store mapped integers as 'int'
                        self.graph.ep[prop_name] = prop
                    else:
                        prop = self.graph.ep[prop_name]
                    
                    # Collect mapped_values
                    prop_values_list.append(mapped_values)
                    eprops.append(prop)
                else:
                    print(f"Unsupported property type for edge property '{prop_name}': {prop_type}")
                    pass  # Extend as needed
        
        # Add edges with numerical properties
        if prop_values_list:
            # Stack edge_array with property values
            edge_list_with_props = np.column_stack((edge_array, *prop_values_list))
            self.graph.add_edge_list(edge_list_with_props, eprops=eprops)
        else:
            # Add edges without numerical properties
            self.graph.add_edge_list(edge_array)
    
    def list_vertex_properties(self):
        """
        List all vertex properties with their types.
        
        Returns:
            pd.DataFrame: DataFrame containing property names and their types.
        """
        prop_names = list(self.graph.vp.keys())
        prop_types = [str(self.graph.vp[prop].value_type()) for prop in prop_names]
        return pd.DataFrame({'Property Name': prop_names, 'Type': prop_types})
    
    def list_edge_properties(self):
        """
        List all edge properties with their types.
        
        Returns:
            pd.DataFrame: DataFrame containing property names and their types.
        """
        prop_names = list(self.graph.ep.keys())
        prop_types = [str(self.graph.ep[prop].value_type()) for prop in prop_names]
        return pd.DataFrame({'Property Name': prop_names, 'Type': prop_types})
    
    def summary(self):
        """
        Provide a summary of the graph including number of vertices, edges, and properties.
        
        Returns:
            dict: Summary information.
        """
        num_vertices = self.graph.num_vertices()
        num_edges = self.graph.num_edges()
        num_vertex_props = len(self.graph.vp)
        num_edge_props = len(self.graph.ep)
        vertex_props = list(self.graph.vp.keys())
        edge_props = list(self.graph.ep.keys())
        return {
            'Number of Vertices': num_vertices,
            'Number of Edges': num_edges,
            'Number of Vertex Properties': num_vertex_props,
            'Number of Edge Properties': num_edge_props,
            'Vertex Properties': vertex_props,
            'Edge Properties': edge_props
        }
    
    def get_vertex_by_encoding_tuple(self, layer_code, node_id_int):
        """
        Retrieve a vertex by its custom ID tuple (layer_code, node_id_int).
        
        Parameters:
            layer_code (int): Integer code of the layer.
            node_id_int (int): Integer code of the node ID.
        
        Returns:
            graph_tool.Vertex or None: The corresponding vertex or None if not found.
        """
        id_tuple = (layer_code, node_id_int)
        v_index = self.custom_id_to_vertex_index.get(id_tuple)
        if v_index is not None:
            return self.graph.vertex(v_index)
        else:
            return None
        
    def get_vertex_by_name_tuple(self, layer_name: str, node_id_str: str) -> Any:
        """
        Retrieve a vertex by its original name using (layer_name, node_id_str) tuples.
        
        Parameters:
            layer_name (str): The name of the layer.
            node_id_str (str): The string identifier of the node.
        
        Returns:
            graph_tool.Vertex or None: The corresponding vertex or None if not found.
        
        Raises:
            KeyError: If the layer name or node ID string does not exist.
        """
        # Retrieve the layer code
        layer_code = self.layer_name_to_code.get(layer_name)
        if layer_code is None:
            raise KeyError(f"Layer name '{layer_name}' not found.")
        
        # Retrieve the node ID integer
        node_id_int = self.node_id_str_to_int.get(node_id_str)
        if node_id_int is None:
            raise KeyError(f"Node ID string '{node_id_str}' not found.")
        
        # Construct the custom ID tuple
        id_tuple = (layer_code, node_id_int)
        
        # Retrieve the vertex index
        vertex_index = self.custom_id_to_vertex_index.get(id_tuple)
        if vertex_index is None:
            return None  # Vertex not found
        
        # Return the vertex object
        return self.graph.vertex(vertex_index)
    
    def get_vertex_property(self, layer_code, node_id_int, prop_name):
        """
        Get the value of a property for a vertex identified by custom ID.
        
        Parameters:
            layer_code (int): Integer code of the layer.
            node_id_int (int): Integer code of the node ID.
            prop_name (str): Name of the property.
        
        Returns:
            any or None: The property value or None if not found.
        """
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None and prop_name in self.graph.vp:
            return self.graph.vp[prop_name][v]
        return None
    
    def set_vertex_property(self, layer_code, node_id_int, prop_name, value):
        """
        Set the value of a property for a vertex identified by custom ID.
        
        Parameters:
            layer_code (int): Integer code of the layer.
            node_id_int (int): Integer code of the node ID.
            prop_name (str): Name of the property.
            value (any): Value to set.
        """
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None:
            if prop_name not in self.graph.vp:
                # Infer property type and create new property
                prop_type = self._infer_property_type(value)
                prop = self.graph.new_vertex_property(prop_type)
                self.graph.vp[prop_name] = prop
            self.graph.vp[prop_name][v] = value
        else:
            print(f"Vertex with ID ({layer_code}, {node_id_int}) not found.")
    
    def view_node_properties(self, layer_code, node_id_int):
        """
        View all properties of a specific node, including decoded layer and node_id.
        
        Parameters:
            layer_code (int): Integer code of the layer.
            node_id_int (int): Integer code of the node ID.
        
        Returns:
            dict: Dictionary of property names and their values.
        """
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is None:
            print(f"Vertex with ID ({layer_code}, {node_id_int}) not found.")
            return {}
        
        properties = {}
        for prop_name in self.graph.vp.keys():
            prop_value = self.graph.vp[prop_name][v]
            # Check if this property was mapped from categorical
            if prop_name in self.vertex_categorical_mappings:
                prop_value = self.vertex_categorical_mappings[prop_name]['int_to_str'].get(prop_value, f"Unknown ({prop_value})")
            properties[prop_name] = prop_value
        
        # Decode layer and node_id
        decoded_layer = self.layer_code_to_name.get(layer_code, f"Unknown Layer ({layer_code})")
        decoded_node_id = self.node_id_int_to_str.get(node_id_int, f"Unknown Node ID ({node_id_int})")
        properties['decoded_layer'] = decoded_layer
        properties['decoded_node_id'] = decoded_node_id
        
        return properties
    
    def view_node_properties_by_names(self, layer_name, node_id_str, verbose=False):
        """
        View all properties of a specific node using layer name and node ID string.
        
        Parameters:
            layer_name (str): Name of the layer.
            node_id_str (str): Node ID as a string.
            verbose (bool, optional): If True, prints the properties.
        
        Returns:
            dict: Dictionary of property names and their values.
        """
        # Check if the layer name exists
        if layer_name in self.layer_name_to_code:
            layer_code = self.layer_name_to_code[layer_name]
        else:
            print(f"Layer '{layer_name}' not found.")
            return {}
        
        # Check if the node ID string exists
        if node_id_str in self.node_id_str_to_int:
            node_id_int = self.node_id_str_to_int[node_id_str]
        else:
            print(f"Node ID '{node_id_str}' not found.")
            return {}
        
        # Retrieve node properties
        properties = self.view_node_properties(layer_code, node_id_int)
        
        if verbose:
            print(f"\nProperties for node (Layer: '{layer_name}', Node ID: '{node_id_str}'):")
            for prop, value in properties.items():
                print(f"  {prop}: {value}")
        
        return properties
    
    def get_edge_property(self, source_layer_code, source_node_id_int, target_layer_code, target_node_id_int, prop_name):
        """
        Get the value of a property for an edge identified by source and target IDs.
        
        Parameters:
            source_layer_code (int): Integer code of the source layer.
            source_node_id_int (int): Integer code of the source node ID.
            target_layer_code (int): Integer code of the target layer.
            target_node_id_int (int): Integer code of the target node ID.
            prop_name (str): Name of the property.
        
        Returns:
            any or None: The property value or None if not found.
        """
        source_vertex = self.get_vertex_by_encoding_tuple(source_layer_code, source_node_id_int)
        target_vertex = self.get_vertex_by_encoding_tuple(target_layer_code, target_node_id_int)
        if source_vertex is None or target_vertex is None:
            print("Source or target vertex not found.")
            return None
        e = self.graph.edge(source_vertex, target_vertex)
        if e is not None and prop_name in self.graph.ep:
            return self.graph.ep[prop_name][e]
        return None
    
    def get_edge_property_by_names(self, source_layer_name, source_node_id_str, target_layer_name, target_node_id_str, prop_name=None, verbose=False):
        """
        Get the value(s) of property/properties for an edge identified by source and target names.
        
        Parameters:
            source_layer_name (str): Layer name for the source node.
            source_node_id_str (str): Node ID string for the source node.
            target_layer_name (str): Layer name for the target node.
            target_node_id_str (str): Node ID string for the target node.
            prop_name (str, optional): Name of the property. If None, all properties are returned.
            verbose (bool, optional): If True, prints the properties.
        
        Returns:
            any or dict: The property value if prop_name is provided, or a dictionary of all properties if prop_name is None.
        """
        # Map layer names and node IDs to integer codes
        if source_layer_name in self.layer_name_to_code:
            source_layer_code = self.layer_name_to_code[source_layer_name]
        else:
            print(f"Source layer '{source_layer_name}' not found.")
            return None
        
        if source_node_id_str in self.node_id_str_to_int:
            source_node_id_int = self.node_id_str_to_int[source_node_id_str]
        else:
            print(f"Source node ID '{source_node_id_str}' not found.")
            return None
        
        if target_layer_name in self.layer_name_to_code:
            target_layer_code = self.layer_name_to_code[target_layer_name]
        else:
            print(f"Target layer '{target_layer_name}' not found.")
            return None
        
        if target_node_id_str in self.node_id_str_to_int:
            target_node_id_int = self.node_id_str_to_int[target_node_id_str]
        else:
            print(f"Target node ID '{target_node_id_str}' not found.")
            return None
        
        # Retrieve the edge
        source_vertex = self.get_vertex_by_encoding_tuple(source_layer_code, source_node_id_int)
        target_vertex = self.get_vertex_by_encoding_tuple(target_layer_code, target_node_id_int)
        
        if source_vertex is None or target_vertex is None:
            print("Source or target vertex not found.")
            return None
        
        e = self.graph.edge(source_vertex, target_vertex)
        if e is None:
            print(f"No edge exists between ({source_layer_name}, '{source_node_id_str}') and ({target_layer_name}, '{target_node_id_str}').")
            return None
        
        if prop_name:
            if prop_name in self.graph.ep:
                value = self.graph.ep[prop_name][e]
                # Check if this property was mapped from categorical
                if prop_name in self.edge_categorical_mappings:
                    value = self.edge_categorical_mappings[prop_name]['int_to_str'].get(value, f"Unknown ({value})")
                if verbose:
                    print(f"\nEdge Property '{prop_name}' between ({source_layer_name}, '{source_node_id_str}') and ({target_layer_name}, '{target_node_id_str}'): {value}")
                return value
            else:
                print(f"Edge property '{prop_name}' not found.")
                return None
        else:
            # Return all properties for the edge
            edge_properties = {}
            for prop in self.graph.ep.keys():
                val = self.graph.ep[prop][e]
                if prop in self.edge_categorical_mappings:
                    val = self.edge_categorical_mappings[prop]['int_to_str'].get(val, f"Unknown ({val})")
                edge_properties[prop] = val
            if verbose:
                print(f"\nAll Edge Properties between ({source_layer_name}, '{source_node_id_str}') and ({target_layer_name}, '{target_node_id_str}'):")
                for prop, value in edge_properties.items():
                    print(f"  {prop}: {value}")
            return edge_properties
        
    def view_layer(self, layer_name):
        """
        Return a subgraph view of a specific layer.

        Parameters:
            layer_name (str): The name of the layer to filter.

        Returns:
            GraphView: A subgraph containing only the vertices from the specified layer.
        """
        if layer_name not in self.layer_name_to_code:
            raise ValueError(f"Layer '{layer_name}' does not exist.")
        
        layer_code = self.layer_name_to_code[layer_name]
        return GraphView(self.graph, vfilt=lambda v: self.graph.vp['layer_hash'][v] == layer_code)
    
    def filter_view_by_property(self, prop_name, target_value, comparison="=="):
        """
        Creates a filtered graph view including only nodes where the specified property
        meets the specified comparison with the target value.

        Parameters:
            prop_name (str): The name of the property to filter by.
            target_value (any): The value to filter for.
            comparison (str): The comparison operator as a string (e.g., "==", "!=", "<", ">").

        Returns:
            GraphView: A subgraph view with the filtered vertices.
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

        # Handle categorical properties by mapping target_value to its integer code
        if prop_name in self.vertex_categorical_mappings:
            str_to_int = self.vertex_categorical_mappings[prop_name]['str_to_int']
            if target_value in str_to_int:
                target_value_mapped = str_to_int[target_value]
            else:
                raise ValueError(f"Target value '{target_value}' not found in categorical mapping for property '{prop_name}'.")
        else:
            target_value_mapped = target_value

        # Create a filter mask based on the property and comparison
        def filter_func(v):
            prop_val = self.graph.vp[prop_name][v]
            return compare_func(prop_val, target_value_mapped)

        # Create and return a filtered GraphView
        return GraphView(self.graph, vfilt=filter_func)
    
    def extract_subgraph_with_paths(self, root_layer_name, root_node_id_str, nodes_of_interest_ids, direction='upstream'):
        """
        Extracts a subgraph that includes the nodes of interest, the root,
        and any intermediate nodes along the paths from each node of interest according to the specified direction.
        Direction can be 'upstream', 'downstream', or 'both'.

        Parameters:
            root_layer_name (str): The layer name of the root node.
            root_node_id_str (str): The node ID string of the root node.
            nodes_of_interest_ids (list of tuples): List of (layer_name, node_id_str) tuples for nodes of interest.
            direction (str): The direction of traversal: 'upstream', 'downstream', or 'both'.

        Returns:
            GraphView: A subgraph containing relevant nodes and edges.
        """

        # Get root vertex
        root_vertex = self.get_vertex_by_encoding_tuple(
            layer_code=self.layer_name_to_code.get(root_layer_name),
            node_id_int=self.node_id_str_to_int.get(root_node_id_str)
        )
        if root_vertex is None:
            raise ValueError(f"Root node ({root_layer_name}, {root_node_id_str}) not found.")

        # Get vertex objects for nodes of interest
        nodes_of_interest_vertices = set()
        for layer_name, node_id_str in nodes_of_interest_ids:
            v = self.get_vertex_by_encoding_tuple(
                layer_code=self.layer_name_to_code.get(layer_name),
                node_id_int=self.node_id_str_to_int.get(node_id_str)
            )
            if v is not None:
                nodes_of_interest_vertices.add(v)
            else:
                print(f"Node ({layer_name}, {node_id_str}) not found and will be skipped.")

        # Initialize global vertex and edge filters
        vfilt = self.graph.new_vertex_property('bool')
        vfilt.a[:] = False
        efilt = self.graph.new_edge_property('bool')
        efilt.a[:] = False

        if direction in ['upstream', 'both']:
            # Upstream traversal (towards ancestors)
            self._bfs_traversal(nodes_of_interest_vertices, vfilt, efilt, mode='upstream')
            # Include the root node in upstream traversal
            vfilt[root_vertex] = True

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

        Parameters:
            seed_vertices (set of Vertex): Seed vertex objects to start traversal.
            vfilt (VertexPropertyMap): Vertex property map to update.
            efilt (EdgePropertyMap): Edge property map to update.
            mode (str): 'upstream' for ancestors, 'downstream' for descendants.
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
        

    def search(
        self, 
        #start_layer_name: str, #TODO: implement as option
        #start_node_id_str: str, #TODO: implement as option
        start_node_idx: int = 0, #The index of the node to start the search from.
        max_dist: int = 5, 
        direction: str = 'downstream', 
        node_text_prop: str = 'node_label',  # Default to the new property
        show_plot: bool = True, 
        **kwargs
    ) -> GraphView:
        """
        Generalized function to perform upstream, downstream, or both-directional search on a directed graph.

        Parameters:
            TODO: start_layer_name (str): The layer name of the node to start the search from.
            TODO: start_node_id_str (str): The node ID string of the node to start the search from.
            start_node_idx (int): The index of the node to start the search from.
            max_dist (int): Maximum distance (in number of hops) to search.
            direction (str): 'downstream', 'upstream', or 'both' to search in both directions.
            node_text_prop (str): Vertex property to use for node labels.
            show_plot (bool): Whether to display the plot.

        Returns:
            GraphView: A filtered subgraph containing the nodes within the given distance in the specified direction.
        """
        g = self.graph
        MAX_DIST = max_dist

        # Step 1: Select the starting node
        start_vertex = g.vertex(start_node_idx)
        # TODO: implement alternative using name and ID

        # Step 2: Handle direction (upstream, downstream, or both)
        if direction == 'upstream':
            # Create a reversed view of the graph for upstream search
            g_reversed = GraphView(g, reversed=True)
            distances = shortest_distance(g_reversed, source=start_vertex, max_dist=MAX_DIST)

        elif direction == 'downstream':
            # Use the graph as is for downstream search
            distances = shortest_distance(g, source=start_vertex, max_dist=MAX_DIST)

        elif direction == 'both':
            # Perform both upstream and downstream searches separately
            # Upstream search with reversed graph
            g_upstream = GraphView(g, reversed=True)
            distances_upstream = shortest_distance(g_upstream, source=start_vertex, max_dist=MAX_DIST)

            # Downstream search
            distances_downstream = shortest_distance(g, source=start_vertex, max_dist=MAX_DIST)

            # Merge distances (take minimum distance if reachable in both directions)
            distances = {}
            for v in g.vertices():
                dist_up = distances_upstream.get(v, float('inf'))
                dist_down = distances_downstream.get(v, float('inf'))
                min_dist = min(dist_up, dist_down)
                if min_dist <= MAX_DIST:
                    distances[v] = min_dist

        else:
            raise ValueError("Invalid direction. Choose 'upstream', 'downstream', or 'both'.")

        # Step 3: Filter the graph to only include nodes within the specified distance
        result_filter = GraphView(g, vfilt=lambda v: distances[v] <= MAX_DIST and distances[v] < float('inf'))

        # Output details
        print(f"{direction.capitalize()} graph from node {start_vertex} contains {result_filter.num_vertices()} vertices and {result_filter.num_edges()} edges.")

        # Optionally draw the filtered graph
        if show_plot:
            if node_text_prop in self.graph.vp:
                vertex_text_prop = self.graph.vp[node_text_prop]
            else:
                # Handle cases where node_text_prop is not a valid property
                vertex_text_prop = self.graph.new_vertex_property('string')
                for v in result_filter.vertices():
                    vertex_text_prop[v] = str(int(v))
            
            graph_draw(result_filter, vertex_text=vertex_text_prop, **kwargs)

        # # Optionally draw the filtered graph
        # if show_plot:
        #     vertex_text_prop = result_filter.vertex_properties[node_text] if node_text != None else result_filter.vertex_index
        #     graph_draw(result_filter, vertex_text=vertex_text_prop, **kwargs)

        return result_filter
    
    def extract_subgraph_with_label_component(self, root_layer_name, root_node_id_str, nodes_of_interest_ids, direction='downstream'):
        """
        Extracts a subgraph including all nodes reachable from nodes of interest 
        in the specified direction ('upstream', 'downstream', or 'both').

        Parameters:
            root_layer_name (str): The layer name of the root node.
            root_node_id_str (str): The node ID string of the root node.
            nodes_of_interest_ids (list of tuples): List of (layer_name, node_id_str) tuples for nodes of interest.
            direction (str): The direction of traversal ('upstream', 'downstream', or 'both').

        Returns:
            GraphView: A subgraph containing relevant nodes and edges.
        """
        from graph_tool.topology import label_out_component

        # Initialize an empty vertex filter property map with False values
        vfilt = self.graph.new_vertex_property('bool')
        vfilt.a[:] = False

        # Get root vertex
        root_vertex = self.get_vertex_by_encoding_tuple(
            layer_code=self.layer_name_to_code.get(root_layer_name),
            node_id_int=self.node_id_str_to_int.get(root_node_id_str)
        )
        if root_vertex is None:
            raise ValueError(f"Root node ({root_layer_name}, {root_node_id_str}) not found.")

        # Get vertex objects for nodes of interest
        nodes_of_interest_vertices = []
        for layer_name, node_id_str in nodes_of_interest_ids:
            v = self.get_vertex_by_encoding_tuple(
                layer_code=self.layer_name_to_code.get(layer_name),
                node_id_int=self.node_id_str_to_int.get(node_id_str)
            )
            if v is not None:
                nodes_of_interest_vertices.append(v)
            else:
                print(f"Node ({layer_name}, {node_id_str}) not found and will be skipped.")

        # Downstream component
        if direction in ['downstream', 'both']:
            for node in nodes_of_interest_vertices:
                component = label_out_component(self.graph, node)
                vfilt.a = vfilt.a | component.a  # Logical OR to combine component labels

        # Upstream component by using a reversed graph view
        if direction in ['upstream', 'both']:
            reversed_graph = GraphView(self.graph, reversed=True)
            for node in nodes_of_interest_vertices:
                component = label_out_component(reversed_graph, node)
                vfilt.a = vfilt.a | component.a  # Logical OR to combine component labels

        # Include the root node explicitly
        vfilt[root_vertex] = True

        # Create the filtered subgraph with the combined vertex filter
        subgraph = GraphView(self.graph, vfilt=vfilt)
        return subgraph
    
    def inspect_properties(self, layer_name, node_id_str, verbose=True):
        """
        Inspect all properties of a specific node, decoding categorical properties.

        Parameters:
            layer_name (str): The layer name of the node.
            node_id_str (str): The node ID string of the node.
            verbose (bool): If True, prints the properties.

        Returns:
            dict: Dictionary of property names and their values.
        """
        # Retrieve properties using existing method
        properties = self.view_node_properties_by_names(layer_name, node_id_str, verbose=False)
        
        if verbose:
            print(f"Properties for node ({layer_name}, {node_id_str}):")
            for prop, value in properties.items():
                print(f"  {prop}: {value}")
        
        return properties
    
    def get_root_nodes(self):
        """
        Identifies root nodes in the graph (nodes with no incoming edges).

        Returns:
            list: List of vertex objects that are root nodes.
        """
        return [v for v in self.graph.vertices() if v.in_degree() == 0]
    
    def create_node_label_property(self, prop_name: str = 'node_label') -> None:
        """
        Creates a new vertex property that combines layer names and node IDs for informative labeling.
        
        Parameters:
            prop_name (str): The name of the new vertex property to create. Defaults to 'node_label'.
        """
        if prop_name in self.graph.vp:
            print(f"Vertex property '{prop_name}' already exists.")
            return
        
        # Create a new string property
        node_label = self.graph.new_vertex_property('string')
        
        # Efficiently generate labels using list comprehension
        labels = [
            f"{self.layer_code_to_name.get(self.graph.vp['layer_hash'][v], f'Unknown Layer ({self.graph.vp['layer_hash'][v]})')}:" +
            f"{self.node_id_int_to_str.get(self.graph.vp['node_id_hash'][v], f'Unknown ID ({self.graph.vp['node_id_hash'][v]})')}"
            for v in self.graph.vertices()
        ]
        
        # Assign the labels to the property
        for v, label in zip(self.graph.vertices(), labels):
            node_label[v] = label
        
        # Assign the new property to the graph
        self.graph.vp[prop_name] = node_label
        print(f"Vertex property '{prop_name}' created successfully.")

    def create_human_readable_property(
        self, 
        encoded_prop_type: str,  # 'v' for vertex, 'e' for edge
        encoded_prop_name: str, 
        mapping_dict: Dict[int, str], 
        new_prop_name: str, 
        default_label: str = 'Unknown'
    ) -> None:
        """
        Creates a new property by mapping encoded integers to human-readable strings.

        Parameters:
            encoded_prop_type (str): Type of the property ('v' for vertex, 'e' for edge).
            encoded_prop_name (str): Name of the existing encoded property.
            mapping_dict (Dict[int, str]): Dictionary mapping encoded integers to strings.
            new_prop_name (str): Name of the new human-readable property to create.
            default_label (str, optional): Default label for unmapped integers. Defaults to 'Unknown'.
        
        Raises:
            ValueError: If the encoded_prop_type is neither 'v' nor 'e'.
            KeyError: If the encoded_prop_name does not exist in the graph.

        Example usage:
        # Assuming you have already populated layer_code_to_name mapping
        onion.create_human_readable_property(
            encoded_prop_type='v', 
            encoded_prop_name='layer_hash', 
            mapping_dict=onion.layer_code_to_name, 
            new_prop_name='layer_name'
        )
        """
        if encoded_prop_type not in ['v', 'e']:
            raise ValueError("encoded_prop_type must be 'v' for vertex or 'e' for edge.")
        
        if (encoded_prop_type, encoded_prop_name) not in self.graph.properties():
            raise KeyError(f"{encoded_prop_type.upper()} property '{encoded_prop_name}' does not exist.")
        
        # Determine the property type
        if encoded_prop_type == 'v':
            prop = self.graph.vp[encoded_prop_name]
        else:
            prop = self.graph.ep[encoded_prop_name]
        
        # Create a new string property
        human_readable_prop = self.graph.new_property('string')
        
        # Efficiently generate labels using list comprehension
        if encoded_prop_type == 'v':
            items = self.graph.vertices()
        else:
            items = self.graph.edges()
        
        labels = [
            mapping_dict.get(int(prop[item]), default_label) for item in items
        ]
        
        # Assign the labels to the new property
        for item, label in zip(items, labels):
            human_readable_prop[item] = label
        
        # Assign the new property to the graph
        if encoded_prop_type == 'v':
            self.graph.vp[new_prop_name] = human_readable_prop
        else:
            self.graph.ep[new_prop_name] = human_readable_prop
        
        print(f"{encoded_prop_type.upper()} property '{new_prop_name}' created successfully.")

    def create_all_human_readable_properties(self) -> None:
        """
        Automatically creates human-readable properties for all encoded vertex and edge properties
        based on existing mapping dictionaries.
        """
        # Define a dictionary of encoded properties and their corresponding mapping dictionaries
        # Format: (type, encoded_prop_name, mapping_dict, new_prop_name)
        mapping_definitions = [
            ('v', 'layer_hash', self.layer_code_to_name, 'layer_name'),
            ('v', 'node_id_hash', self.node_id_int_to_str, 'node_id_str'),
            # Add more mappings as needed
            # Example for edge properties:
            # ('e', 'edge_prop_1', edge_prop_1_code_to_name, 'edge_prop_1_name'),

            ######## TODO #########
            # TODO: MORE TO ADD: TODO
            ######## TODO #########
        ]
        
        for prop_type, encoded_prop, mapping_dict, new_prop in mapping_definitions:
            try:
                self.create_human_readable_property(
                    encoded_prop_type=prop_type,
                    encoded_prop_name=encoded_prop,
                    mapping_dict=mapping_dict,
                    new_prop_name=new_prop
                )
            except KeyError as e:
                print(f"Mapping failed for {prop_type.upper()} property '{encoded_prop}': {e}")
            except ValueError as e:
                print(f"Invalid property type for '{prop_type}': {e}")

    @property
    def node_map(self):
        """
        Creates and returns a mapping from (layer_name, node_id_str) tuples to vertex indices.
        
        Returns:
            dict: A dictionary mapping (layer_name, node_id_str) to vertex index.
        """
        # Initialize node_map if it hasn't been created yet
        if not hasattr(self, '_node_map'):
            self._node_map = {}
            for (layer_code, node_id_int), v_idx in self.custom_id_to_vertex_index.items():
                layer_name = self.layer_code_to_name.get(layer_code, f"Unknown Layer ({layer_code})")
                node_id_str = self.node_id_int_to_str.get(node_id_int, f"Unknown ID ({node_id_int})")
                self._node_map[(layer_name, node_id_str)] = v_idx
        return self._node_map
    
