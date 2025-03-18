import streamlit as st
import pandas as pd
import time
import os
from data_loader import load_data
from network_generator import get_network
from visualizations import create_combined_visualization

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(page_title="Hospital Acquired Infections (HAI) Dashboard", 
                       layout="wide", 
                       page_icon="🔬")
    
    # App header
    st.title("Hospital Acquired Infections (HAI) Dashboard")

    # Create sidebar controls for sample size before data loading
    with st.sidebar:
        st.header("Sample Size")
        
        max_sample_points = st.slider(
            "Maximum Facilities per Type", 
            min_value=50, 
            max_value=5000, 
            value=1000,
            step=50,
            help="Controls how many facilities to show. Higher values show more detail but may reduce performance."
        )
        
        # Add an 'all' option using a checkbox
        show_all_points = st.checkbox(
            "Show All Available Facilities", 
            value=False,
            help="Warning: Showing all facilities may significantly reduce performance"
        )
        
        # If show_all_points is checked, set max_sample_points to 0 (meaning "all")
        if show_all_points:
            max_sample_points = 0
            st.info("Showing all facilities. This may take longer to render.")
        
        # Performance options
        st.header("Performance Settings")
        use_optimized = st.checkbox(
            "Use Optimized Network Builder", 
            value=True,
            help="Uses spatial indexing to speed up network calculations"
        )
        
        # Precomputed network options
        use_precomputed = st.checkbox(
            "Use Precomputed Network (if available)", 
            value=True,
            help="Load a previously computed network from disk"
        )
        
        # Hidden save_precomputed variable (set to False by default)
        save_precomputed = False
        
        # Show precomputed network status only if found
        precomputed_file = "precomputed_network.pkl"
        if os.path.exists(precomputed_file):
            st.success(f"Precomputed network found: {precomputed_file}")
    
    # Load data with sample size parameter
    with st.spinner("Loading facility data..."):
        ltc_data, hospital_data = load_data(max_points_per_type=max_sample_points)
    
    # Create network
    with st.spinner("Building network model..."):
        start_time = time.time()
        
        # Get the network, with precomputation options
        network = get_network(
            ltc_data, 
            hospital_data,
            use_optimized=use_optimized,
            use_precomputed=use_precomputed,
            save_precomputed=save_precomputed
        )
            
        # Calculate build time but don't display it
        build_time = time.time() - start_time
        
        # Removed success message about build time
    
    # Create sidebar filters
    with st.sidebar:
        st.header("Visualization Settings")
        
        # Facility filters
        st.subheader("Facility Types")
        show_ltc = st.checkbox("Show Long-Term Care Facilities", value=True)
        show_hospitals = st.checkbox("Show Hospitals", value=True)
        
        # Size filter
        st.subheader("Facility Size")
        min_beds = st.slider("Minimum Beds", 
                            min_value=0, 
                            max_value=500, 
                            value=20,
                            step=10)
        
        # Connection filter
        st.subheader("Connection Strength")
        min_connection = st.slider("Minimum Connection Strength",
                                  min_value=0.0,
                                  max_value=1.0,
                                  value=0.2,
                                  step=0.1)
        
        # Add dataset statistics to sidebar
        st.header("Dataset Statistics")
        
        if show_ltc:
            filtered_ltc = ltc_data[ltc_data["beds"] >= min_beds]
            st.subheader("Long-Term Care Facilities")
            st.info(f"Showing {len(filtered_ltc)} of {len(ltc_data)} facilities")
            st.info(f"Total beds: {filtered_ltc['beds'].sum():,}")
        
        if show_hospitals:
            filtered_hosp = hospital_data[hospital_data["beds"] >= min_beds]
            st.subheader("Hospitals")
            st.info(f"Showing {len(filtered_hosp)} of {len(hospital_data)} facilities")
            st.info(f"Total beds: {filtered_hosp['beds'].sum():,}")
        
        # Network statistics
        st.header("Network Statistics")
        st.info(f"Nodes: {network.number_of_nodes()}")
        st.info(f"Connections: {network.number_of_edges()}")
    
    # Display network visualization
    figure = create_combined_visualization(
        network, 
        ltc_data, 
        hospital_data, 
        min_beds=min_beds,
        min_connection_strength=min_connection,
        show_ltc=show_ltc, 
        show_hospitals=show_hospitals
    )
    st.plotly_chart(figure, use_container_width=True)
    


if __name__ == "__main__":
    main() 