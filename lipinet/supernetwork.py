import pandas as pd
import numpy as np
from graph_tool.all import Graph
import pickle


# A super fast alternative to the previous implementation. 
# Less flexible, but capable of creating networks from huge datasets (e.g. 10m+ nodes) 
# in only minutes. So long as you have the memory to hold the dfs and they are relatively clean.
# Relies on the use of mapping numericals and categoricals, plus some graph-tool tricks.

# Future dev:
# - polar integration


class SuperOnion:
    def __init__(self, directed=True):
        self.graph = Graph(directed=directed)
        self.id_to_index = {}  # Map from custom ID tuple (layer_code, node_id_int) to vertex index
        self.index_to_id = {}  # Map from vertex index to custom ID tuple (layer_code, node_id_int)
        self.layer_reverse_mapping = {}  # Map from layer codes to layer names
        self.node_id_reverse_mapping = {}  # Map from node_id_int to node_id strings
        self.layer_str_to_int = {}  # Map from layer names to integer codes
        self.node_id_str_to_int = {}  # Map from node_id strings to integer codes
        
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
        if layer_name in self.layer_str_to_int:
            return self.layer_str_to_int[layer_name]
        else:
            # Assign a new integer code
            layer_code = len(self.layer_str_to_int)
            self.layer_str_to_int[layer_name] = layer_code
            self.layer_reverse_mapping[layer_code] = layer_name
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
            self.node_id_reverse_mapping[node_id_int] = node_id_str
            return node_id_int
    
    def add_vertices_from_dataframe(self, df_nodes, id_col, layer_col, property_cols=None, drop_na=True, fill_na_with=None):
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
        self.id_to_index.update(zip(custom_ids, new_indices))
        self.index_to_id.update(zip(new_indices, custom_ids))
        
        # Assign 'layer_hash' and 'node_id_hash' properties in bulk
        self.graph.vp['layer_hash'].a[starting_index:] = df_nodes['layer_int'].values
        self.graph.vp['node_id_hash'].a[starting_index:] = df_nodes['node_id_int'].values
        
        # Assign additional properties
        if property_cols:
            for prop_name in property_cols:
                prop_values = df_nodes[prop_name].values
                sample_value = prop_values[0]
                prop_type = self._infer_property_type(sample_value)
                
                if prop_type in ['int', 'float']:
                    if prop_name not in self.graph.vp:
                        prop = self.graph.new_vertex_property(prop_type)
                        self.graph.vp[prop_name] = prop
                    else:
                        prop = self.graph.vp[prop_name]
                    
                    # Assign values in bulk
                    prop.a[starting_index:] = prop_values
                elif prop_type == 'string':
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
    
    def add_edges_from_dataframe(self, df_edges, source_id_col, source_layer_col, target_id_col, target_layer_col, property_cols=None, drop_na=True, fill_na_with=None):
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
        source_indices = [self.id_to_index.get(id_tuple) for id_tuple in source_ids]
        target_indices = [self.id_to_index.get(id_tuple) for id_tuple in target_ids]
        
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
                
                if prop_type in ['int', 'float']:
                    if prop_name not in self.graph.ep:
                        prop = self.graph.new_edge_property(prop_type)
                        self.graph.ep[prop_name] = prop
                    else:
                        prop = self.graph.ep[prop_name]
                    
                    # Collect prop_values
                    prop_values_list.append(prop_values)
                    eprops.append(prop)
                elif prop_type == 'string':
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
    
    def get_vertex_by_custom_id(self, layer_code, node_id_int):
        """
        Retrieve a vertex by its custom ID tuple (layer_code, node_id_int).
        
        Parameters:
            layer_code (int): Integer code of the layer.
            node_id_int (int): Integer code of the node ID.
        
        Returns:
            graph_tool.Vertex or None: The corresponding vertex or None if not found.
        """
        id_tuple = (layer_code, node_id_int)
        v_index = self.id_to_index.get(id_tuple)
        if v_index is not None:
            return self.graph.vertex(v_index)
        else:
            return None
    
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
        v = self.get_vertex_by_custom_id(layer_code, node_id_int)
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
        v = self.get_vertex_by_custom_id(layer_code, node_id_int)
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
        v = self.get_vertex_by_custom_id(layer_code, node_id_int)
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
        decoded_layer = self.layer_reverse_mapping.get(layer_code, f"Unknown Layer ({layer_code})")
        decoded_node_id = self.node_id_reverse_mapping.get(node_id_int, f"Unknown Node ID ({node_id_int})")
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
        if layer_name in self.layer_str_to_int:
            layer_code = self.layer_str_to_int[layer_name]
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
        source_vertex = self.get_vertex_by_custom_id(source_layer_code, source_node_id_int)
        target_vertex = self.get_vertex_by_custom_id(target_layer_code, target_node_id_int)
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
        if source_layer_name in self.layer_str_to_int:
            source_layer_code = self.layer_str_to_int[source_layer_name]
        else:
            print(f"Source layer '{source_layer_name}' not found.")
            return None
        
        if source_node_id_str in self.node_id_str_to_int:
            source_node_id_int = self.node_id_str_to_int[source_node_id_str]
        else:
            print(f"Source node ID '{source_node_id_str}' not found.")
            return None
        
        if target_layer_name in self.layer_str_to_int:
            target_layer_code = self.layer_str_to_int[target_layer_name]
        else:
            print(f"Target layer '{target_layer_name}' not found.")
            return None
        
        if target_node_id_str in self.node_id_str_to_int:
            target_node_id_int = self.node_id_str_to_int[target_node_id_str]
        else:
            print(f"Target node ID '{target_node_id_str}' not found.")
            return None
        
        # Retrieve the edge
        source_vertex = self.get_vertex_by_custom_id(source_layer_code, source_node_id_int)
        target_vertex = self.get_vertex_by_custom_id(target_layer_code, target_node_id_int)
        
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