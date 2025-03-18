import streamlit as st
import pandas as pd
from data_loader import load_data
from network_generator import create_infection_network
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
    
    # Load data with sample size parameter
    with st.spinner("Loading facility data..."):
        ltc_data, hospital_data = load_data(max_points_per_type=max_sample_points)
    
    # Create network
    with st.spinner("Building network model..."):
        network = create_infection_network(ltc_data, hospital_data)
    
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