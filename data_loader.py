import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any

@st.cache_data
def load_data(max_points_per_type: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess healthcare facility data.
    
    Args:
        max_points_per_type: Maximum number of facilities to include per type
                            (0 means include all)
    
    Returns:
        Tuple of (ltc_data, hospital_data): Cleaned pandas DataFrames
    """
    try:
        # Dataset configurations
        configs = {
            'ltc': {
                'file': "data/facility_2021.csv",
                'columns': {"nhlong": "longitude", "nhlat": "latitude", "totbeds": "beds", "prov0475": "name"},
                'required': ["state", "county", "totbeds", "nhlong", "nhlat"],
                'default_name': "LTC Facility"
            },
            'hospital': {
                'file': "data/us_hospital_locations.csv",
                'columns': {"STATE": "state", "COUNTY": "county", "BEDS": "beds", 
                           "LONGITUDE": "longitude", "LATITUDE": "latitude", "NAME": "name"},
                'required': ["STATE", "COUNTY", "BEDS", "LONGITUDE", "LATITUDE"],
                'default_name': "Hospital"
            }
        }
        
        # Process datasets
        ltc_data = _process_dataset(configs['ltc'], max_points_per_type)
        hospital_data = _process_dataset(configs['hospital'], max_points_per_type)
            
        return ltc_data, hospital_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def _process_dataset(config: Dict[str, Any], max_points: int) -> pd.DataFrame:
    """Process a facility dataset using the provided configuration"""
    # Load data
    df = pd.read_csv(config['file'])
    
    # Select columns available in the dataset
    available_columns = set(df.columns).union(config['required'])
    rename_map = {col: new_name for col, new_name in config['columns'].items() 
                 if col in df.columns}
    
    # Extract required data
    required_cols = [col for col in config['required'] if col in df.columns]
    df = df[required_cols + list(set(rename_map.keys()) - set(required_cols))].dropna(subset=required_cols)
    
    # Rename columns
    df = df.rename(columns=rename_map)
    
    # Convert bed count to numeric and filter invalid
    bed_col = rename_map.get(
        next((col for col in config['required'] if 'BED' in col.upper()), None), 
        'beds'
    )
    df[bed_col] = pd.to_numeric(df[bed_col], errors="coerce")
    df = df[df[bed_col] > 0]
    
    # Handle facility name
    if "name" not in df.columns or df["name"].isna().any():
        if "name" not in df.columns:
            df["name"] = config['default_name']
        else:
            df["name"] = df["name"].fillna(config['default_name'])
    
    # Sample if needed
    if max_points > 0 and len(df) > max_points:
        df = df.sample(max_points, random_state=42)
        
    return df 