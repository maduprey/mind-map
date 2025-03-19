import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import random
from typing import Dict, Any, List, Tuple

# Note: This function is currently unused but kept for potential future use
def create_facilities_map(ltc_data, hospital_data, min_beds, show_ltc=True, show_hospitals=True):
    """Create a map visualization of healthcare facilities"""
    # Filter based on minimum beds
    filtered_ltc = ltc_data[ltc_data["beds"] >= min_beds] if show_ltc else pd.DataFrame()
    filtered_hosp = hospital_data[hospital_data["beds"] >= min_beds] if show_hospitals else pd.DataFrame()
    
    # Create figure
    fig = go.Figure()
    
    # Helper function to add facility markers
    def add_facility_markers(df, color, name, bed_label):
        if not df.empty:
            fig.add_trace(go.Scattergeo(
                lon=df["longitude"],
                lat=df["latitude"],
                text=df.apply(lambda row: f"<b>{row['name']}</b><br>{bed_label}: {row['beds']}", axis=1),
                mode="markers",
                marker=dict(
                    size=df["beds"].apply(lambda x: min(max(5, x/20), 30)),
                    color=color,
                    opacity=0.8,
                    line=dict(width=1, color=color)
                ),
                name=name,
                hoverinfo="text"
            ))
    
    # Add facilities to map
    if show_ltc:
        add_facility_markers(filtered_ltc, "#FFD700", "Long-term Care Facilities", "LTC Beds")
    
    if show_hospitals:
        add_facility_markers(filtered_hosp, "#0068c9", "Hospitals", "Hospital Beds")
    
    # Set layout
    fig.update_layout(
        title="US Healthcare Facilities",
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#f0f2f6",
            countrycolor="#c6cad4",
            coastlinecolor="#c6cad4",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#e5ecf6",
            showrivers=True,
            rivercolor="#e5ecf6",
        ),
        legend=dict(x=0, y=0, bgcolor="rgba(255, 255, 255, 0.7)"),
        height=600,
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="sans-serif")
    )
    
    return fig

# Note: This function is currently unused but kept for potential future use
def create_network_visualization(G, min_beds, min_connection_strength, show_ltc=True, show_hospitals=True):
    """Create a network graph visualization of infection connections"""
    # Apply filters
    nodes_to_keep = [node for node in G.nodes() if G.nodes[node]["beds"] >= min_beds]
    
    if not show_ltc:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "LTC"]
    if not show_hospitals:
        nodes_to_keep = [node for node in nodes_to_keep if G.nodes[node]["type"] != "HOSP"]
    
    # Handle empty selection
    if not nodes_to_keep:
        fig = go.Figure()
        fig.add_annotation(
            text="No facilities selected. Please enable at least one facility type.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        return fig
    
    # Create subgraph and filter by connection strength
    subgraph = G.subgraph(nodes_to_keep)
    edges_to_keep = [(u, v) for u, v in subgraph.edges() if subgraph[u][v]["weight"] >= min_connection_strength]
    
    # Create filtered graph
    edge_subgraph = nx.Graph()
    for node in subgraph.nodes():
        edge_subgraph.add_node(node, **subgraph.nodes[node])
    for u, v in edges_to_keep:
        edge_subgraph.add_edge(u, v, weight=subgraph[u][v]["weight"])
    
    # Handle different visualization cases
    connected_components = list(nx.connected_components(edge_subgraph))
    filtered_components = [comp for comp in connected_components if len(comp) > 1]
    
    # Handle case with no valid components
    if not filtered_components:
        if (show_ltc and not show_hospitals) or (show_hospitals and not show_ltc):
            # Show nodes without connections for single facility type
            single_nodes = list(edge_subgraph.nodes())
            if single_nodes:
                component_graph = edge_subgraph
                largest_component = single_nodes
            else:
                # No nodes meet criteria
                fig = go.Figure()
                fig.add_annotation(
                    text="No facilities meet the current filter criteria",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
                )
                return fig
        else:
            # No connections meet criteria
            fig = go.Figure()
            fig.add_annotation(
                text="No connections meet the current filter criteria",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
            )
            return fig
    else:
        # Use largest connected component
        largest_component = max(filtered_components, key=len)
        component_graph = edge_subgraph.subgraph(largest_component)
    
    # Create layout for nodes
    pos = create_node_layout(component_graph)
    
    # Create traces
    data = []
    
    # Add edge trace if there are edges
    if component_graph.edges():
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color="#888"),
            hoverinfo="none", mode="lines"
        )
        
        for edge in component_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)
        
        data.append(edge_trace)
    
    # Add node traces
    node_traces = create_node_traces(component_graph, pos)
    data.extend(node_traces)
    
    # Create figure
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            titlefont=dict(size=16),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="sans-serif")
        )
    )
    
    # Add annotation for single facility type
    if not component_graph.edges() and ((show_ltc and not show_hospitals) or (show_hospitals and not show_ltc)):
        facility_type = "Long-term Care Facilities" if show_ltc else "Hospitals"
        fig.add_annotation(
            text=f"Showing approximate geographic distribution of {facility_type}",
            xref="paper", yref="paper", x=0.5, y=0.99, showarrow=False,
            font=dict(size=14, color="#555555"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cccccc", borderwidth=1, borderpad=4
        )
    
    return fig

def create_combined_visualization(G, ltc_data, hospital_data, min_beds, min_connection_strength, show_ltc=True, show_hospitals=True):
    """
    Create a combined map showing facilities and their transmission connections
    
    Args:
        G: NetworkX graph with facility nodes and connection edges
        ltc_data: DataFrame with long-term care facility data
        hospital_data: DataFrame with hospital data
        min_beds: Minimum number of beds for a facility to be shown
        min_connection_strength: Minimum connection weight to display
        show_ltc: Whether to show long-term care facilities
        show_hospitals: Whether to show hospitals
        
    Returns:
        Plotly figure object with the visualization
    """
    # Filter data and create figure
    filtered_ltc = ltc_data[ltc_data["beds"] >= min_beds] if show_ltc else pd.DataFrame()
    filtered_hosp = hospital_data[hospital_data["beds"] >= min_beds] if show_hospitals else pd.DataFrame()
    fig = go.Figure()
    
    # Filter network nodes - use list comprehension with conditions
    nodes_to_keep = [
        node for node in G.nodes() 
        if G.nodes[node]["beds"] >= min_beds and (
            (G.nodes[node]["type"] == "LTC" and show_ltc) or
            (G.nodes[node]["type"] == "HOSP" and show_hospitals)
        )
    ]
    
    # Handle empty selection
    if not nodes_to_keep:
        fig.add_annotation(
            text="No facilities selected. Please enable at least one facility type.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20)
        )
        return fig
    
    # Create subgraph and filter connections
    subgraph = G.subgraph(nodes_to_keep)
    edges_to_keep = [(u, v) for u, v in subgraph.edges() if subgraph[u][v]["weight"] >= min_connection_strength]
    
    # Build visualization elements
    add_connection_strength_legend(fig, edges_to_keep)
    for u, v in edges_to_keep:
        add_connection_line(fig, subgraph, u, v)
    add_facilities_to_map(fig, filtered_ltc, filtered_hosp, show_ltc, show_hospitals)
    
    # Apply layout configuration
    fig.update_layout(
        geo={
            'scope': 'usa',
            'showland': True,
            'landcolor': "#f0f2f6",
            'countrycolor': "#c6cad4",
            'coastlinecolor': "#c6cad4",
            'projection_type': "albers usa",
            'showlakes': True,
            'lakecolor': "#e5ecf6",
            'showrivers': True,
            'rivercolor': "#e5ecf6",
        },
        legend=dict(x=0, y=0, bgcolor="rgba(255, 255, 255, 0.7)"),
        height=800,
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="sans-serif")
    )
    
    # Add connection information
    add_connection_info(fig, edges_to_keep)
    
    return fig

# Helper functions
def create_node_layout(graph):
    """Create positions for network visualization"""
    if not graph.edges():
        # Geographic layout for disconnected nodes
        pos = {}
        for node in graph.nodes():
            node_info = graph.nodes[node]
            if 'lat' in node_info and 'lon' in node_info:
                try:
                    lon = float(node_info['lon'])
                    lat = float(node_info['lat'])
                    x = (lon + 97.5) / 55.0 * 2 - 1
                    y = (lat - 37.5) / 25.0 * 2 - 1
                    pos[node] = (x, y)
                except (ValueError, TypeError):
                    pos[node] = (random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8))
            else:
                pos[node] = (random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8))
                
        # Add jitter to prevent overlaps
        for node in pos:
            x, y = pos[node]
            pos[node] = (x + random.uniform(-0.05, 0.05), y + random.uniform(-0.05, 0.05))
    else:
        # Spring layout for connected graphs
        pos = nx.spring_layout(graph, seed=42)
    
    return pos

def create_node_traces(graph, pos):
    """Create node traces for hospitals and LTCs"""
    node_trace_ltc = go.Scatter(
        x=[], y=[], text=[], mode="markers", hoverinfo="text",
        marker=dict(color="#FFD700", size=[], line=dict(width=2)),
        name="Long-term Care Facilities"
    )
    
    node_trace_hosp = go.Scatter(
        x=[], y=[], text=[], mode="markers", hoverinfo="text",
        marker=dict(color="#0068c9", size=[], line=dict(width=2)),
        name="Hospitals"
    )
    
    traces = []
    ltc_present = False
    hosp_present = False
    
    for node in graph.nodes():
        x, y = pos[node]
        node_info = graph.nodes[node]
        facility_type = 'LTC' if node_info['type'] == 'LTC' else 'Hospital'
        facility_name = node_info.get('name', f"{facility_type}")
        state = node_info.get('state', '')
        
        # Create base hover text
        hover_text = f"<b>{facility_name}</b><br>Type: {facility_type}<br>State: {state}<br>Beds: {node_info['beds']}"
        
        # Add additional details for LTC facilities
        if facility_type == 'LTC':
            # Dictionary of attributes to include in hover text
            hover_attributes = {
                'occpct': {'label': 'Occupancy', 'format': '{:.1f}%'},
                'avg_dailycensus': {'label': 'Avg Daily Census', 'format': '{:.1f}'},
                'adm_bed': {'label': 'Admissions per Bed', 'format': '{:.2f}'},
                'dchprd_pbj': {'label': 'Discharge Rate', 'format': '{:.2f}'},
                'obs_successfuldc': {'label': 'Successful Discharge', 'format': '{:.4f}'},
                'obs_rehosprate': {'label': 'Rehospitalization Rate', 'format': '{:.4f}'}
            }
            
            # Loop through and add each attribute if available
            for attr_name, config in hover_attributes.items():
                try:
                    if attr_name in node_info and pd.notna(node_info[attr_name]):
                        value = float(node_info[attr_name])
                        formatted_value = config['format'].format(value)
                        hover_text += f"<br>{config['label']}: {formatted_value}"
                except (ValueError, TypeError, KeyError) as e:
                    print(f"Error adding {attr_name} to hover text: {e}")
        
        connections = len(list(graph.neighbors(node)))
        if connections > 0:
            hover_text += f"<br>Connections: {connections}"
        
        node_size = min(max(10, node_info['beds']/10), 50)
        
        if node_info["type"] == "LTC":
            ltc_present = True
            node_trace_ltc["x"] = node_trace_ltc["x"] + (x,)
            node_trace_ltc["y"] = node_trace_ltc["y"] + (y,)
            node_trace_ltc["text"] = node_trace_ltc["text"] + (hover_text,)
            node_trace_ltc["marker"]["size"] = node_trace_ltc["marker"]["size"] + (node_size,)
        else:
            hosp_present = True
            node_trace_hosp["x"] = node_trace_hosp["x"] + (x,)
            node_trace_hosp["y"] = node_trace_hosp["y"] + (y,)
            node_trace_hosp["text"] = node_trace_hosp["text"] + (hover_text,)
            node_trace_hosp["marker"]["size"] = node_trace_hosp["marker"]["size"] + (node_size,)
    
    if ltc_present:
        traces.append(node_trace_ltc)
    if hosp_present:
        traces.append(node_trace_hosp)
    
    return traces

def add_connection_strength_legend(fig, edges):
    """Add a legend for connection strength to the figure."""
    if not edges:
        return
    
    # Create legend items with consistent line width but varying opacity
    line_width = 1  # Reduced from 2 to make lines narrower
    
    legend_items = [
        {"name": "Weak (0.0-0.3)", "color": "rgba(80,80,80,0.25)", "width": line_width},
        {"name": "Medium (0.3-0.6)", "color": "rgba(80,80,80,0.45)", "width": line_width},
        {"name": "Strong (0.6-1.0)", "color": "rgba(80,80,80,0.65)", "width": line_width},
    ]
    
    # Add legend traces (invisible lines with legend entries)
    for item in legend_items:
        fig.add_trace(
            go.Scattergeo(
                lon=[None], lat=[None],  # No visible points
                mode="lines",
                line=dict(width=item["width"], color=item["color"]),
                name=item["name"],
                showlegend=True,
                legendgroup="connections",
            )
        )

def add_connection_line(fig, G, u, v):
    """Add a connection line between two facilities on the map."""
    # Get node data
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    weight = G[u][v]["weight"]
    
    # Use a fixed line width but vary color opacity based on connection strength
    line_width = 1  # Reduced from 2 to make lines narrower
    
    # Set opacity based on connection strength - reduced all values to make less dark
    if weight > 0.6:  # Strong
        opacity = 0.65  # Reduced from 0.9
    elif weight > 0.3:  # Medium
        opacity = 0.45  # Reduced from 0.6
    else:  # Weak
        opacity = 0.25  # Reduced from 0.3
    
    # Create color with varying opacity
    color = f"rgba(80,80,80,{opacity})"
    
    # Add the line
    fig.add_trace(
        go.Scattergeo(
            lon=[u_data["lon"], v_data["lon"]],
            lat=[u_data["lat"], v_data["lat"]],
            mode="lines",
            line=dict(width=line_width, color=color),
            showlegend=False,
            hoverinfo="skip",
        )
    )

def add_facilities_to_map(fig, ltc_data, hospital_data, show_ltc, show_hospitals):
    """Add facility markers to the map."""
    # Add LTC facilities
    if show_ltc and not ltc_data.empty:
        # Check which columns are actually available in the dataframe
        has_occpct = 'occpct' in ltc_data.columns
        has_avg_dailycensus = 'avg_dailycensus' in ltc_data.columns
        has_adm_bed = 'adm_bed' in ltc_data.columns
        has_dchprd_pbj = 'dchprd_pbj' in ltc_data.columns
        has_obs_successfuldc = 'obs_successfuldc' in ltc_data.columns
        has_obs_rehosprate = 'obs_rehosprate' in ltc_data.columns
        
        # Print column info for debugging
        print(f"LTC dataframe columns: {ltc_data.columns.tolist()}")
        print(f"Column dtypes: {ltc_data.dtypes}")
        
        # Convert numeric columns to float if they exist
        if has_occpct:
            ltc_data['occpct'] = pd.to_numeric(ltc_data['occpct'], errors='coerce')
        if has_avg_dailycensus:
            ltc_data['avg_dailycensus'] = pd.to_numeric(ltc_data['avg_dailycensus'], errors='coerce')
        if has_adm_bed:
            ltc_data['adm_bed'] = pd.to_numeric(ltc_data['adm_bed'], errors='coerce')
        if has_dchprd_pbj:
            ltc_data['dchprd_pbj'] = pd.to_numeric(ltc_data['dchprd_pbj'], errors='coerce')
        if has_obs_successfuldc:
            ltc_data['obs_successfuldc'] = pd.to_numeric(ltc_data['obs_successfuldc'], errors='coerce')
        if has_obs_rehosprate:
            ltc_data['obs_rehosprate'] = pd.to_numeric(ltc_data['obs_rehosprate'], errors='coerce')
        
        # Create hover text based on available columns
        def create_hover_text(row):
            text = f"<b>{row['name']}</b><br>Beds: {row['beds']}"
            
            try:
                if has_occpct and pd.notna(row['occpct']):
                    text += f"<br>Occupancy: {float(row['occpct']):.1f}%"
            except (ValueError, TypeError):
                pass
                
            try:
                if has_avg_dailycensus and pd.notna(row['avg_dailycensus']):
                    text += f"<br>Avg Daily Census: {float(row['avg_dailycensus']):.1f}"
            except (ValueError, TypeError):
                pass
                
            try:
                if has_adm_bed and pd.notna(row['adm_bed']):
                    text += f"<br>Admissions per Bed: {float(row['adm_bed']):.2f}"
            except (ValueError, TypeError):
                pass
                
            try:
                if has_dchprd_pbj and pd.notna(row['dchprd_pbj']):
                    text += f"<br>Discharge Rate: {float(row['dchprd_pbj']):.2f}"
            except (ValueError, TypeError):
                pass
                
            try:
                if has_obs_successfuldc and pd.notna(row['obs_successfuldc']):
                    text += f"<br>Successful Discharge: {float(row['obs_successfuldc']):.4f}"
            except (ValueError, TypeError):
                pass
                
            try:
                if has_obs_rehosprate and pd.notna(row['obs_rehosprate']):
                    text += f"<br>Rehospitalization Rate: {float(row['obs_rehosprate']):.4f}"
            except (ValueError, TypeError):
                pass
                
            return text
                
        fig.add_trace(
            go.Scattergeo(
                lon=ltc_data["longitude"],
                lat=ltc_data["latitude"],
                text=ltc_data.apply(create_hover_text, axis=1),
                mode="markers",
                marker=dict(
                    size=ltc_data["beds"].apply(lambda x: min(max(x/20, 5), 15)),
                    color="rgba(255, 215, 0, 0.8)",
                    line=dict(width=1, color="rgba(204, 172, 0, 1)"),
                    symbol="circle",
                ),
                name="Long-term Care",
                hovertemplate="%{text}<extra></extra>",
            )
        )
    
    # Add hospital facilities
    if show_hospitals and not hospital_data.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=hospital_data["longitude"],
                lat=hospital_data["latitude"],
                text=hospital_data.apply(lambda row: f"<b>{row['name']}</b><br>Beds: {row['beds']}", axis=1),
                mode="markers",
                marker=dict(
                    size=hospital_data["beds"].apply(lambda x: min(max(x/30, 5), 20)),
                    color="rgba(66, 135, 245, 0.8)",
                    line=dict(width=1, color="rgba(40, 86, 166, 1)"),
                    symbol="circle",
                ),
                name="Hospitals",
                hovertemplate="%{text}<extra></extra>",
            )
        )

def add_connection_info(fig, edges):
    """Add connection information text to the figure."""
    if not edges:
        fig.add_annotation(
            text="No connections meet the current strength threshold.",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
    else:
        fig.add_annotation(
            text=f"Showing {len(edges)} connections",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=14, color="gray"),
            align="left",
        ) 