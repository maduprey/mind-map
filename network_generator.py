import networkx as nx
import numpy as np
import pandas as pd
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

def create_infection_network(
    ltc_data: pd.DataFrame, 
    hospital_data: pd.DataFrame, 
    max_distance_km: float = 100.0
) -> nx.Graph:
    """
    Create a network graph representing potential infection transmission between facilities.
    
    Args:
        ltc_data: DataFrame with long-term care facility data
        hospital_data: DataFrame with hospital data
        max_distance_km: Maximum distance to consider for connections
        
    Returns:
        NetworkX graph with facilities as nodes and transmission paths as edges
    """
    # Create empty graph
    G = nx.Graph()
    
    # Combine facility data
    all_facilities = []
    
    # Add LTC facilities to graph
    for idx, facility in ltc_data.iterrows():
        node_id = f"LTC_{idx}"  # Use DataFrame index instead of 'id' column
        G.add_node(
            node_id,
            id=str(idx),  # Store index as string id
            name=facility['name'],
            lat=facility['latitude'],
            lon=facility['longitude'],
            beds=facility['beds'],
            type="LTC"
        )
        all_facilities.append((node_id, facility))
    
    # Add hospitals to graph
    for idx, facility in hospital_data.iterrows():
        node_id = f"HOSP_{idx}"  # Use DataFrame index instead of 'id' column
        G.add_node(
            node_id,
            id=str(idx),  # Store index as string id
            name=facility['name'],
            lat=facility['latitude'],
            lon=facility['longitude'],
            beds=facility['beds'],
            type="HOSP"
        )
        all_facilities.append((node_id, facility))
    
    # Calculate connections between facilities
    for i, (node1_id, facility1) in enumerate(all_facilities):
        # Only need to compare with facilities not yet processed (avoid duplicates)
        for node2_id, facility2 in all_facilities[i+1:]:
            # Calculate distance between facilities
            try:
                lat1, lon1 = facility1['latitude'], facility1['longitude']
                lat2, lon2 = facility2['latitude'], facility2['longitude']
                distance_km = calculate_distance(lat1, lon1, lat2, lon2)
                
                # Only create connections within maximum distance and when distance is valid
                if distance_km <= max_distance_km and distance_km != float('inf'):
                    # Calculate connection strength
                    strength = calculate_connection_strength(
                        distance_km, 
                        facility1['beds'], 
                        facility2['beds']
                    )
                    
                    # Add edge to graph with connection data
                    G.add_edge(
                        node1_id, 
                        node2_id, 
                        weight=strength, 
                        distance=distance_km
                    )
            except (KeyError, TypeError, ValueError) as e:
                # Skip this pair if there's any issue with the data
                continue
    
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