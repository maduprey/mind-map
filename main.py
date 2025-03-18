import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Set page configuration
# Note: Theme can be configured in .streamlit/config.toml or through the UI settings
st.set_page_config(
    page_title="Hospital Acquired Infections Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title
st.title("Hospital Acquired Infections (HAI) Dashboard")

# Load the data
@st.cache_data
def load_data(max_points_per_type=1000):
    # Load long-term care facilities data
    try:
        ltc_data = pd.read_csv("data/facility_2021.csv")
        # Load hospital data
        hospital_data = pd.read_csv("data/us_hospital_locations.csv")
        
        # Comment out the success messages to hide them
        # st.sidebar.success(f"Successfully loaded LTC data with {len(ltc_data)} rows")
        # st.sidebar.success(f"Successfully loaded hospital data with {len(hospital_data)} rows")
        
        # Clean the data
        # For LTC, extract the facility name from prov2720
        ltc_cols = ["state", "county", "totbeds", "nhlong", "nhlat"]
        # Check if prov2720 exists (facility name)
        if "prov2720" in ltc_data.columns:
            ltc_cols.append("prov2720")
            
        ltc_data = ltc_data[ltc_cols].dropna(subset=["state", "county", "totbeds", "nhlong", "nhlat"])
        ltc_data = ltc_data.rename(columns={
            "nhlong": "longitude", 
            "nhlat": "latitude", 
            "totbeds": "beds",
            "prov2720": "name" if "prov2720" in ltc_data.columns else None
        })
        
        # For hospitals, use NAME column for facility name
        hosp_cols = ["STATE", "COUNTY", "BEDS", "LONGITUDE", "LATITUDE"]
        if "NAME" in hospital_data.columns:
            hosp_cols.append("NAME")
            
        hospital_data = hospital_data[hosp_cols].dropna(subset=["STATE", "COUNTY", "BEDS", "LONGITUDE", "LATITUDE"])
        hospital_data = hospital_data.rename(columns={
            "STATE": "state", 
            "COUNTY": "county", 
            "BEDS": "beds", 
            "LONGITUDE": "longitude", 
            "LATITUDE": "latitude",
            "NAME": "name" if "NAME" in hospital_data.columns else None
        })
        
        # Convert beds to numeric
        ltc_data["beds"] = pd.to_numeric(ltc_data["beds"], errors="coerce")
        hospital_data["beds"] = pd.to_numeric(hospital_data["beds"], errors="coerce")
        
        # Filter out any invalid beds data
        ltc_data = ltc_data[ltc_data["beds"] > 0]
        hospital_data = hospital_data[hospital_data["beds"] > 0]
        
        # Add default name if name column doesn't exist or has NaN values
        if "name" not in ltc_data.columns:
            ltc_data["name"] = "LTC Facility"
        else:
            # Fix pandas warning by avoiding inplace operations on chain
            ltc_data = ltc_data.copy()
            ltc_data["name"] = ltc_data["name"].fillna("LTC Facility")
            
        if "name" not in hospital_data.columns:
            hospital_data["name"] = "Hospital"
        else:
            # Fix pandas warning by avoiding inplace operations on chain
            hospital_data = hospital_data.copy()
            hospital_data["name"] = hospital_data["name"].fillna("Hospital")
        
        # Sample the data based on the max_points_per_type parameter
        # Use the full dataset if max_points_per_type is set to 0 (meaning "all")
        if max_points_per_type > 0:
            if len(ltc_data) > max_points_per_type:
                ltc_data = ltc_data.sample(max_points_per_type, random_state=42)
            if len(hospital_data) > max_points_per_type:
                hospital_data = hospital_data.sample(max_points_per_type, random_state=42)
            
        return ltc_data, hospital_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty dataframes in case of error
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def create_infection_network(ltc_data, hospital_data, max_nodes_per_type=200):
    """
    Create a simulated infection network between hospitals and LTCs.
    The strength of connections is based on geographic proximity and facility size.
    """
    G = nx.Graph()
    
    # Add LTC facilities as nodes
    for idx, row in ltc_data.iterrows():
        G.add_node(f"LTC_{idx}", 
                   type="LTC", 
                   beds=row["beds"], 
                   lat=row["latitude"], 
                   lon=row["longitude"],
                   state=row["state"],
                   name=row["name"])
    
    # Add hospitals as nodes
    for idx, row in hospital_data.iterrows():
        G.add_node(f"HOSP_{idx}", 
                   type="HOSP", 
                   beds=row["beds"], 
                   lat=row["latitude"], 
                   lon=row["longitude"],
                   state=row["state"],
                   name=row["name"])
    
    # Create edges between facilities
    # This is a simplified model where:
    # 1. We only connect facilities in the same state
    # 2. Connection strength is simulated with random values but influenced by bed count
    
    # Sample LTC nodes (for performance reasons - full dataset would be too large)
    ltc_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "LTC"]
    hosp_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "HOSP"]
    
    # Sample nodes based on the max_nodes_per_type parameter
    # Use the full dataset if max_nodes_per_type is set to 0 (meaning "all")
    if max_nodes_per_type > 0:
        if len(ltc_nodes) > max_nodes_per_type:
            ltc_nodes = random.sample(ltc_nodes, max_nodes_per_type)
        if len(hosp_nodes) > max_nodes_per_type:
            hosp_nodes = random.sample(hosp_nodes, max_nodes_per_type)
    
    with st.spinner("Creating infection network simulation..."):
        # Create edges
        for ltc in ltc_nodes:
            ltc_state = G.nodes[ltc]["state"]
            ltc_beds = G.nodes[ltc]["beds"]
            
            for hosp in hosp_nodes:
                if G.nodes[hosp]["state"] == ltc_state:
                    hosp_beds = G.nodes[hosp]["beds"]
                    
                    # Simple model: connection strength is influenced by facility sizes
                    # and a random factor (to simulate real-world variability)
                    conn_strength = (ltc_beds * hosp_beds / 10000) * random.uniform(0.5, 1.5)
                    
                    # Only add connections above a minimum threshold
                    if conn_strength > 0.1:
                        G.add_edge(ltc, hosp, weight=conn_strength)
    
    return G

def create_facilities_map(ltc_data, hospital_data, min_beds, show_ltc=True, show_hospitals=True):
    """Create a map visualization of LTCs and hospitals"""
    # Filter based on min beds
    filtered_ltc = ltc_data[ltc_data["beds"] >= min_beds]
    filtered_hosp = hospital_data[hospital_data["beds"] >= min_beds]
    
    # Create a figure with two traces
    fig = go.Figure()
    
    # Add LTC facilities if enabled
    if show_ltc:
        fig.add_trace(go.Scattergeo(
            lon=filtered_ltc["longitude"],
            lat=filtered_ltc["latitude"],
            text=filtered_ltc.apply(lambda row: f"<b>{row['name']}</b><br>LTC Beds: {row['beds']}", axis=1),
            mode="markers",
            marker=dict(
                size=filtered_ltc["beds"].apply(lambda x: min(max(5, x/20), 30)),
                color="#0068c9",  # Streamlit blue
                opacity=0.8,
                line=dict(width=1, color="#0068c9")
            ),
            name="Long-term Care Facilities",
            hoverinfo="text"
        ))
    
    # Add hospitals if enabled
    if show_hospitals:
        fig.add_trace(go.Scattergeo(
            lon=filtered_hosp["longitude"],
            lat=filtered_hosp["latitude"],
            text=filtered_hosp.apply(lambda row: f"<b>{row['name']}</b><br>Hospital Beds: {row['beds']}", axis=1),
            mode="markers",
            marker=dict(
                size=filtered_hosp["beds"].apply(lambda x: min(max(5, x/20), 30)),
                color="#ff4b4b",  # Streamlit red
                opacity=0.8,
                line=dict(width=1, color="#ff4b4b")
            ),
            name="Hospitals",
            hoverinfo="text"
        ))
    
    # Update layout with Streamlit-themed colors
    fig.update_layout(
        title="US Healthcare Facilities",
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#f0f2f6",  # Light gray background like Streamlit
            countrycolor="#c6cad4",  # Slightly darker gray for borders
            coastlinecolor="#c6cad4",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#e5ecf6",  # Light blue for lakes
            showrivers=True,
            rivercolor="#e5ecf6",  # Light blue for rivers
        ),
        legend=dict(
            x=0,
            y=0,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ),
        height=600,
        paper_bgcolor="#ffffff",  # White background
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="sans-serif")
    )
    
    return fig

def create_network_visualization(G, min_beds, min_connection_strength, show_ltc=True, show_hospitals=True):
    """Create a network graph visualization of infection connections"""
    # Create a subgraph with nodes that meet the bed count filter
    nodes_to_keep = [node for node in G.nodes() if G.nodes[node]["beds"] >= min_beds]
    
    # Apply facility type filter
    if not show_ltc:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "LTC"]
    if not show_hospitals:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "HOSP"]
    
    # If no nodes to show, return an empty figure
    if not nodes_to_keep:
        fig = go.Figure()
        fig.add_annotation(
            text="No facilities selected. Please enable at least one facility type.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    subgraph = G.subgraph(nodes_to_keep)
    
    # Filter edges based on connection strength
    edges_to_keep = [(u, v) for u, v in subgraph.edges() if subgraph[u][v]["weight"] >= min_connection_strength]
    edge_subgraph = nx.Graph()
    
    # Add nodes
    for node in subgraph.nodes():
        edge_subgraph.add_node(node, **subgraph.nodes[node])
    
    # Add filtered edges
    for u, v in edges_to_keep:
        edge_subgraph.add_edge(u, v, weight=subgraph[u][v]["weight"])
    
    # For a more readable visualization, limit to components with at least 2 nodes
    connected_components = list(nx.connected_components(edge_subgraph))
    filtered_components = [comp for comp in connected_components if len(comp) > 1]
    
    # If there are no valid components after filtering for connection strength
    if not filtered_components:
        # Special case: when only one facility type is selected and no connections exist
        if (show_ltc and not show_hospitals) or (show_hospitals and not show_ltc):
            # Display nodes without connections
            single_nodes = list(edge_subgraph.nodes())
            if single_nodes:
                # Just show the nodes without edges if we have at least one node
                component_graph = edge_subgraph
                largest_component = single_nodes
            else:
                # No nodes meet criteria
                fig = go.Figure()
                fig.add_annotation(
                    text="No facilities meet the current filter criteria",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                return fig
        else:
            # Standard case: no connections meet criteria
            fig = go.Figure()
            fig.add_annotation(
                text="No connections meet the current filter criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
    else:
        # Use the largest connected component for visualization
        largest_component = max(filtered_components, key=len)
        component_graph = edge_subgraph.subgraph(largest_component)
    
    # Create positions for the network graph
    # Use a better layout when there are no edges (single facility type)
    if not component_graph.edges():
        # Use a layout based on geographic coordinates instead of a circle
        pos = {}
        for node in component_graph.nodes():
            # Normalize latitude and longitude to the [-1, 1] range for visualization
            node_info = component_graph.nodes[node]
            # Use geographic position (if available) or random position
            if 'lat' in node_info and 'lon' in node_info:
                # Normalize and scale the coordinates
                # First ensure the coordinates are floats
                try:
                    lon = float(node_info['lon'])
                    lat = float(node_info['lat'])
                    
                    # US longitude roughly spans -125 to -70, latitude spans 25 to 50
                    x = (lon + 97.5) / 55.0 * 2 - 1  # Center and normalize to [-1,1]
                    y = (lat - 37.5) / 25.0 * 2 - 1  # Center and normalize to [-1,1]
                    pos[node] = (x, y)
                except (ValueError, TypeError):
                    # If conversion fails, use random position
                    pos[node] = (random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8))
            else:
                # Fallback to random if geo coordinates aren't available
                pos[node] = (random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8))
                
        # Add a small random jitter to prevent exact overlaps
        for node in pos:
            x, y = pos[node]
            pos[node] = (x + random.uniform(-0.05, 0.05), y + random.uniform(-0.05, 0.05))
    else:
        # Use standard spring layout for connected graphs
        pos = nx.spring_layout(component_graph, seed=42)
    
    # Create a networkx visualization using plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )
    
    # Add edges to the trace - only if the graph has edges
    if component_graph.edges():
        for edge in component_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)
    
    # Create node traces for hospitals and LTCs
    node_trace_ltc = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="#0068c9",  # Use Streamlit blue
            size=[],
            line=dict(width=2)
        ),
        name="Long-term Care Facilities"
    )
    
    node_trace_hosp = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="#ff4b4b",  # Use Streamlit red
            size=[],
            line=dict(width=2)
        ),
        name="Hospitals"
    )
    
    # Add nodes to traces
    ltc_nodes_present = False
    hosp_nodes_present = False
    for node in component_graph.nodes():
        x, y = pos[node]
        node_info = component_graph.nodes[node]
        facility_type = 'LTC' if node_info['type'] == 'LTC' else 'Hospital'
        facility_name = node_info.get('name', f"{facility_type}")
        
        # Enhanced hover text with state information
        state = node_info.get('state', '')
        hover_text = f"<b>{facility_name}</b><br>Type: {facility_type}<br>State: {state}<br>Beds: {node_info['beds']}"
        
        # Get number of connections
        connections = len(list(component_graph.neighbors(node)))
        if connections > 0:
            hover_text += f"<br>Connections: {connections}"
        
        node_size = min(max(10, node_info['beds']/10), 50)
        
        if node_info["type"] == "LTC":
            ltc_nodes_present = True
            node_trace_ltc["x"] = node_trace_ltc["x"] + (x,)
            node_trace_ltc["y"] = node_trace_ltc["y"] + (y,)
            node_trace_ltc["text"] = node_trace_ltc["text"] + (hover_text,)
            node_trace_ltc["marker"]["size"] = node_trace_ltc["marker"]["size"] + (node_size,)
        else:  # Hospital
            hosp_nodes_present = True
            node_trace_hosp["x"] = node_trace_hosp["x"] + (x,)
            node_trace_hosp["y"] = node_trace_hosp["y"] + (y,)
            node_trace_hosp["text"] = node_trace_hosp["text"] + (hover_text,)
            node_trace_hosp["marker"]["size"] = node_trace_hosp["marker"]["size"] + (node_size,)
    
    # Prepare data for the figure, only including present node types
    data = []
    
    # Only add edge trace if there are edges
    if component_graph.edges():
        data.append(edge_trace)
    
    if ltc_nodes_present:
        data.append(node_trace_ltc)
    if hosp_nodes_present:
        data.append(node_trace_hosp)
    
    # Create the figure with Streamlit theme colors
    fig = go.Figure(data=data,
                   layout=go.Layout(
                       title="Candida auris Infection Network",
                       titlefont=dict(size=16),
                       showlegend=True,
                       hovermode="closest",
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       paper_bgcolor="#ffffff",  # White background
                       plot_bgcolor="#ffffff",   # White plot area
                       font=dict(family="sans-serif")
                   ))
    
    # Add an annotation when showing only one facility type
    if not component_graph.edges() and ((show_ltc and not show_hospitals) or (show_hospitals and not show_ltc)):
        facility_type = "Long-term Care Facilities" if show_ltc else "Hospitals"
        fig.add_annotation(
            text=f"Showing approximate geographic distribution of {facility_type}",
            xref="paper", yref="paper",
            x=0.5, y=0.99,
            showarrow=False,
            font=dict(size=14, color="#555555"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4
        )
    
    return fig

def create_combined_visualization(G, ltc_data, hospital_data, min_beds, min_connection_strength, show_ltc=True, show_hospitals=True):
    """Create a combined map showing both facilities and their transmission connections"""
    # Filter based on min beds and facility type
    filtered_ltc = ltc_data[ltc_data["beds"] >= min_beds] if show_ltc else pd.DataFrame()
    filtered_hosp = hospital_data[hospital_data["beds"] >= min_beds] if show_hospitals else pd.DataFrame()
    
    # Create base figure with map
    fig = go.Figure()
    
    # Get nodes from the network that match our filters
    nodes_to_keep = [node for node in G.nodes() if G.nodes[node]["beds"] >= min_beds]
    
    # Apply facility type filter
    if not show_ltc:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "LTC"]
    if not show_hospitals:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "HOSP"]
    
    # If no nodes to show, return an empty figure with message
    if not nodes_to_keep:
        fig.add_annotation(
            text="No facilities selected. Please enable at least one facility type.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create subgraph with filtered nodes
    subgraph = G.subgraph(nodes_to_keep)
    
    # Filter edges based on connection strength
    edges_to_keep = [(u, v) for u, v in subgraph.edges() if subgraph[u][v]["weight"] >= min_connection_strength]
    
    # Add a legend entry for connections (using a dummy trace)
    if edges_to_keep:
        # Create three dummy traces for the legend to show connection strength levels
        fig.add_trace(go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines",
            line=dict(
                width=6,
                color="rgba(60, 60, 60, 0.8)"  # Dark gray for strong connections
            ),
            name="Strong Connection"
        ))
        
        fig.add_trace(go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines",
            line=dict(
                width=4,
                color="rgba(120, 120, 120, 0.8)"  # Medium gray for medium connections
            ),
            name="Medium Connection"
        ))
        
        fig.add_trace(go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="lines",
            line=dict(
                width=2,
                color="rgba(180, 180, 180, 0.8)"  # Light gray for weak connections
            ),
            name="Weak Connection"
        ))
    
    # Plot the edges (connections) between facilities
    for u, v in edges_to_keep:
        # Get the geographic coordinates for both facilities
        u_lat = subgraph.nodes[u]["lat"]
        u_lon = subgraph.nodes[u]["lon"]
        v_lat = subgraph.nodes[v]["lat"]
        v_lon = subgraph.nodes[v]["lon"]
        
        # Calculate connection strength for line width and opacity
        weight = subgraph[u][v]["weight"]
        # Increase the minimum and maximum line width for better visibility
        line_width = min(max(2, weight * 1.5), 8)  # Increased width range (2-8 pixels)
        # Increase minimum opacity for better visibility
        opacity = min(max(0.5, weight/8), 0.9)  # Higher opacity range (0.5-0.9)
        
        # Color code by connection strength
        if weight > 5:
            line_color = "rgba(60, 60, 60, " + str(opacity) + ")"  # Strong: Dark gray
        elif weight > 2:
            line_color = "rgba(120, 120, 120, " + str(opacity) + ")"  # Medium: Medium gray
        else:
            line_color = "rgba(180, 180, 180, " + str(opacity) + ")"  # Weak: Light gray
        
        # Draw line between the facilities with more visible color and opacity
        fig.add_trace(go.Scattergeo(
            lon=[u_lon, v_lon],
            lat=[u_lat, v_lat],
            mode="lines",
            line=dict(
                width=line_width,
                color=line_color
            ),
            opacity=opacity,
            hoverinfo="text",
            hovertext=f"Connection strength: {weight:.2f}",
            showlegend=False
        ))
    
    # Add LTC facilities if enabled
    if show_ltc and not filtered_ltc.empty:
        fig.add_trace(go.Scattergeo(
            lon=filtered_ltc["longitude"],
            lat=filtered_ltc["latitude"],
            text=filtered_ltc.apply(lambda row: f"<b>{row['name']}</b><br>LTC Beds: {row['beds']}", axis=1),
            mode="markers",
            marker=dict(
                size=filtered_ltc["beds"].apply(lambda x: min(max(5, x/20), 30)),
                color="#0068c9",  # Streamlit blue
                opacity=0.8,
                line=dict(width=1, color="#0068c9")
            ),
            name="Long-term Care Facilities",
            hoverinfo="text"
        ))
    
    # Add hospitals if enabled
    if show_hospitals and not filtered_hosp.empty:
        fig.add_trace(go.Scattergeo(
            lon=filtered_hosp["longitude"],
            lat=filtered_hosp["latitude"],
            text=filtered_hosp.apply(lambda row: f"<b>{row['name']}</b><br>Hospital Beds: {row['beds']}", axis=1),
            mode="markers",
            marker=dict(
                size=filtered_hosp["beds"].apply(lambda x: min(max(5, x/20), 30)),
                color="#ff4b4b",  # Streamlit red
                opacity=0.8,
                line=dict(width=1, color="#ff4b4b")
            ),
            name="Hospitals",
            hoverinfo="text"
        ))
    
    # Update layout with Streamlit-themed colors
    fig.update_layout(
        title="Candida auris Transmission Map",
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#f0f2f6",  # Light gray background like Streamlit
            countrycolor="#c6cad4",  # Slightly darker gray for borders
            coastlinecolor="#c6cad4",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#e5ecf6",  # Light blue for lakes
            showrivers=True,
            rivercolor="#e5ecf6",  # Light blue for rivers
        ),
        legend=dict(
            x=0,
            y=0,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ),
        height=800,  # Make this visualization taller
        paper_bgcolor="#ffffff",  # White background
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="sans-serif")
    )
    
    # Add information about connections if they exist
    if edges_to_keep:
        fig.add_annotation(
            text=f"Showing {len(edges_to_keep)} potential transmission paths",
            xref="paper", yref="paper",
            x=0.99, y=0.05,  # Moved up slightly
            showarrow=False,
            xanchor="right",
            font=dict(size=14, color="#555555"),  # Increased font size
            bgcolor="rgba(255, 255, 255, 0.9)",  # More opaque background
            bordercolor="#555555",  # Gray to match connection lines
            borderwidth=2,
            borderpad=6
        )
    else:
        fig.add_annotation(
            text="No connections meet the current filter criteria",
            xref="paper", yref="paper",
            x=0.99, y=0.05,  # Moved up slightly
            showarrow=False,
            xanchor="right",
            font=dict(size=14, color="#555555"),  # Increased font size
            bgcolor="rgba(255, 255, 255, 0.9)",  # More opaque background
            bordercolor="#cccccc",
            borderwidth=2,
            borderpad=6
        )
    
    return fig

def main():
    # Add sidebar with filters
    st.sidebar.title("Filter Options")
    
    # Create a sample size slider
    st.sidebar.markdown("### Sample Size")
    max_sample_points = st.sidebar.slider(
        "Maximum Facilities per Type", 
        min_value=50, 
        max_value=5000, 
        value=2000,  # Default to a reasonable number 
        step=50,
        help="Controls how many facilities to show. Higher values show more detail but may reduce performance."
    )
    
    # Add an 'all' option using a checkbox
    show_all_points = st.sidebar.checkbox(
        "Show All Available Facilities", 
        value=False,
        help="Warning: Showing all facilities may significantly reduce performance"
    )
    
    # If show_all_points is checked, set max_sample_points to 0 (meaning "all")
    if show_all_points:
        max_sample_points = 0
        st.sidebar.info("Showing all facilities. This may take longer to render.")
    
    # Load data with the sample size parameter
    ltc_data, hospital_data = load_data(max_sample_points)
    
    # Create infection network with a fixed node limit for better performance
    # Using a smaller fixed limit (400 nodes maximum) for the network visualization
    network_node_limit = 200  # Fixed limit per facility type for network generation
    infection_network = create_infection_network(ltc_data, hospital_data, network_node_limit)
    
    # Add facility type toggles
    st.sidebar.markdown("### Facility Types")
    show_ltc = st.sidebar.checkbox("Show Long-term Care Facilities", value=True)
    show_hospitals = st.sidebar.checkbox("Show Hospitals", value=True)
    
    # Add a warning if both facility types are turned off
    if not show_ltc and not show_hospitals:
        st.sidebar.warning("⚠️ At least one facility type should be enabled for meaningful visualization")
    
    # Create sliders for filtering
    st.sidebar.markdown("### Filtering Options")
    min_beds = st.sidebar.slider(
        "Minimum Number of Beds", 
        min_value=0, 
        max_value=500, 
        value=0, 
        step=10
    )
    
    connection_strength = st.sidebar.slider(
        "Minimum Connection Strength", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0, 
        step=0.1
    )
    
    # Display statistics based on selected facility types
    st.sidebar.markdown("### Dataset Statistics")
    
    if show_ltc:
        st.sidebar.info(f"LTC Facilities: {len(ltc_data)} total, {len(ltc_data[ltc_data['beds'] >= min_beds])} with {min_beds}+ beds")
        
    if show_hospitals:
        st.sidebar.info(f"Hospitals: {len(hospital_data)} total, {len(hospital_data[hospital_data['beds'] >= min_beds])} with {min_beds}+ beds")
    
    # Show help information
    with st.sidebar.expander("About this dashboard"):
        st.markdown("""
        This dashboard visualizes potential Candida auris transmission between healthcare facilities.
        
        - **Blue dots**: Long-term care facilities
        - **Red dots**: Hospitals
        - The size of dots represents the number of beds
        - Connection strength is simulated based on facility size and location
        - **Gray lines** on the combined map show potential transmission paths
          - Darker, thicker lines = stronger connections
          - Lighter, thinner lines = weaker connections
        
        Use the checkboxes to toggle facility types on/off and the sliders to filter facilities by bed count and connection strength.
        """)
    
    # Create the full-width transmission map visualization
    combined_viz = create_combined_visualization(infection_network, ltc_data, hospital_data, min_beds, connection_strength, show_ltc, show_hospitals)
    
    # Update the height for a larger visualization
    combined_viz.update_layout(height=900)
    
    # Display the visualization full-width
    st.plotly_chart(combined_viz, use_container_width=True)

if __name__ == "__main__":
    main() 