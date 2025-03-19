import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import os
import pickle
from typing import Dict, Tuple, List, Union, Optional

def calculate_connection_strength(
    distance_km: float, 
    facility1_beds: int, 
    facility2_beds: int,
    distance_scale: float = 60.0,    # Decreased to make distance decay slower
    bed_scale: float = 0.4,          # Significantly increased
    max_strength: float = 0.95       # Increased max cap
) -> float:
    """
    Calculate connection strength between two facilities based on distance and bed count.
    
    Args:
        distance_km: Distance between facilities in kilometers
        facility1_beds: Number of beds in first facility
        facility2_beds: Number of beds in second facility
        distance_scale: Scaling factor for distance impact
        bed_scale: Scaling factor for bed count impact
        max_strength: Maximum possible connection strength
        
    Returns:
        Float representing connection strength (0-1)
    """
    try:
        # Convert inputs to appropriate types
        distance_km = float(distance_km)
        facility1_beds = int(facility1_beds)
        facility2_beds = int(facility2_beds)
        
        # Validate inputs
        if distance_km < 0 or facility1_beds < 0 or facility2_beds < 0:
            return 0.0
        
        # Apply a more gradual distance decay
        distance_factor = 1.0 / (1.0 + (distance_km / distance_scale))  # Removed power for more linear decay
        
        # More direct scaling of bed counts with less dampening
        bed_factor = np.sqrt(facility1_beds * facility2_beds) * bed_scale / 100.0
        
        # Basic linear combination with minimal normalization
        strength = distance_factor * (1.0 + bed_factor)
        
        # Cap at max_strength
        return min(max_strength, strength)
    except (ValueError, TypeError):
        return 0.0

@st.cache_data
def create_optimized_infection_network(
    ltc_data: pd.DataFrame, 
    hospital_data: pd.DataFrame, 
    max_distance_km: float = 100.0
) -> nx.Graph:
    """
    Create a network graph using spatial optimization to speed up processing.
    
    This optimized version reduces the number of distance calculations needed
    by using a grid-based spatial index to only compare facilities that are 
    potentially within the maximum distance.
    
    Args:
        ltc_data: DataFrame with long-term care facility data
        hospital_data: DataFrame with hospital data
        max_distance_km: Maximum distance to consider for connections
        
    Returns:
        NetworkX graph with facilities as nodes and transmission paths as edges
    """
    # Create empty graph
    G = nx.Graph()
    
    # Helper function to create nodes
    def add_facilities_to_graph(df, type_prefix):
        facilities = []
        for idx, facility in df.iterrows():
            try:
                node_id = f"{type_prefix}_{idx}"
                lat = float(facility['latitude'])
                lon = float(facility['longitude'])
                beds = int(facility['beds'])
                name = str(facility['name'])
                
                # Create a node with base attributes
                node_attrs = {
                    'id': str(idx),
                    'name': name,
                    'lat': lat,
                    'lon': lon,
                    'beds': beds,
                    'type': type_prefix,
                    # Add grid cell indices for spatial indexing
                    'cell_x': int((lon + 180) / 2),  # 2-degree grid cells
                    'cell_y': int((lat + 90) / 2)
                }
                
                # For LTC facilities, add additional attributes if available
                if type_prefix == "LTC":
                    # Print available columns for debugging
                    print(f"Available columns for LTC facility: {list(facility.index)}")
                    
                    # Create a dictionary of columns to add
                    additional_columns = {
                        'occpct': 'occpct',
                        'avg_dailycensus': 'avg_dailycensus',
                        'adm_bed': 'adm_bed',
                        'dchprd_pbj': 'dchprd_pbj',
                        'obs_successfuldc': 'obs_successfuldc',
                        'obs_rehosprate': 'obs_rehosprate'
                    }
                    
                    # Loop through and add each column safely
                    for col_name, attr_name in additional_columns.items():
                        try:
                            if col_name in facility and pd.notna(facility[col_name]):
                                value = float(facility[col_name])
                                node_attrs[attr_name] = value
                                print(f"Added {attr_name}: {value}")
                            else:
                                print(f"Column {col_name} not available or is NaN")
                        except (ValueError, TypeError) as e:
                            print(f"Error adding {col_name}: {e}")
                    
                    # Print all node attributes for debugging
                    print(f"Node attributes for {node_id}: {node_attrs}")
                
                # Only add facilities with valid coordinates
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    G.add_node(node_id, **node_attrs)
                    facilities.append(node_id)
            except (ValueError, KeyError, TypeError):
                # Skip facilities with invalid data
                continue
        return facilities
    
    # Add facilities as nodes
    ltc_nodes = add_facilities_to_graph(ltc_data, "LTC")
    hosp_nodes = add_facilities_to_graph(hospital_data, "HOSP")
    
    # Create a spatial index (map grid cells to node IDs)
    grid_index = {}
    for node in G.nodes:
        cell_key = (G.nodes[node]['cell_x'], G.nodes[node]['cell_y'])
        if cell_key not in grid_index:
            grid_index[cell_key] = []
        grid_index[cell_key].append(node)
    
    # Compute the maximum grid cell distance to consider
    # 2 degrees is approximately 222km at the equator, less elsewhere
    # So 1 degree is roughly 111km
    grid_distance = int(max_distance_km / 111) + 1
    
    # Process nodes and create edges using spatial indexing
    processed_edges = set()
    
    for node1 in G.nodes:
        cell_x = G.nodes[node1]['cell_x']
        cell_y = G.nodes[node1]['cell_y']
        
        # Only check neighboring cells that could contain nodes within max_distance
        for dx in range(-grid_distance, grid_distance + 1):
            for dy in range(-grid_distance, grid_distance + 1):
                neighbor_cell = (cell_x + dx, cell_y + dy)
                
                if neighbor_cell in grid_index:
                    for node2 in grid_index[neighbor_cell]:
                        # Skip self-connections and already processed pairs
                        edge_key = tuple(sorted([node1, node2]))
                        if node1 != node2 and edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            
                            # Calculate actual distance
                            lat1, lon1 = G.nodes[node1]['lat'], G.nodes[node1]['lon']
                            lat2, lon2 = G.nodes[node2]['lat'], G.nodes[node2]['lon']
                            distance_km = calculate_distance(lat1, lon1, lat2, lon2)
                            
                            # Create edge if within maximum distance
                            if distance_km <= max_distance_km:
                                strength = calculate_connection_strength(
                                    distance_km,
                                    G.nodes[node1]['beds'],
                                    G.nodes[node2]['beds']
                                )
                                
                                G.add_edge(
                                    node1,
                                    node2,
                                    weight=strength,
                                    distance=distance_km
                                )
    
    return G

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance between points in kilometers
    """
    try:
        # Convert coordinates to floats
        lat1, lon1 = float(lat1), float(lon1)
        lat2, lon2 = float(lat2), float(lon2)
        
        # Basic validation of coordinates
        if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90) or not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
            return float('inf')  # Return infinity for invalid coordinates
            
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        R = 6371
        
        return R * c
    except (ValueError, TypeError):
        # Return infinity for any conversion errors
        return float('inf') 

def save_network(G: nx.Graph, filename: str = "precomputed_network.pkl") -> bool:
    """
    Save a precomputed network to disk.
    
    Args:
        G: NetworkX graph to save
        filename: Name of the file to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(G, f)
        return True
    except Exception as e:
        st.error(f"Error saving network: {str(e)}")
        return False

def load_network(filename: str = "precomputed_network.pkl") -> Optional[nx.Graph]:
    """
    Load a precomputed network from disk.
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        NetworkX graph if successful, None otherwise
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                G = pickle.load(f)
            return G
        return None
    except Exception as e:
        st.error(f"Error loading network: {str(e)}")
        return None

@st.cache_data
def get_network(
    ltc_data: pd.DataFrame, 
    hospital_data: pd.DataFrame, 
    use_optimized: bool = True,
    use_precomputed: bool = False,
    save_precomputed: bool = False,
    filename: str = "precomputed_network.pkl",
    max_distance_km: float = 100.0
) -> nx.Graph:
    """
    Get the infection network, using precomputed data if available and requested.
    
    Args:
        ltc_data: DataFrame with long-term care facility data
        hospital_data: DataFrame with hospital data
        use_optimized: Whether to use the optimized network builder
        use_precomputed: Whether to try loading a precomputed network
        save_precomputed: Whether to save the network if computed
        filename: Name of the file to save to/load from
        max_distance_km: Maximum distance to consider for connections
        
    Returns:
        NetworkX graph with facilities as nodes and transmission paths as edges
    """
    # Try to load precomputed network if requested
    if use_precomputed:
        G = load_network(filename)
        if G is not None:
            return G
    
    # Otherwise compute the network
    if use_optimized:
        G = create_optimized_infection_network(ltc_data, hospital_data, max_distance_km)
    else:
        G = create_infection_network(ltc_data, hospital_data, max_distance_km)
    
    # Save the network if requested
    if save_precomputed:
        save_network(G, filename)
    
    return G 