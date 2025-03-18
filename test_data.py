import pandas as pd
import numpy as np

def test_data_loading():
    """Test if we can load and process the data files correctly"""
    try:
        # Load long-term care facilities data
        ltc_data = pd.read_csv("data/facility_2021.csv")
        # Load hospital data
        hospital_data = pd.read_csv("data/us_hospital_locations.csv")
        
        print(f"Successfully loaded LTC data with {len(ltc_data)} rows")
        print(f"Successfully loaded hospital data with {len(hospital_data)} rows")
        
        # Check if required columns exist in LTC data
        ltc_cols = ["state", "county", "totbeds", "nhlong", "nhlat"]
        missing_ltc_cols = [col for col in ltc_cols if col not in ltc_data.columns]
        if missing_ltc_cols:
            print(f"WARNING: Missing columns in LTC data: {missing_ltc_cols}")
        else:
            print("All required columns present in LTC data")
            
            # Check for valid coordinate data
            ltc_sample = ltc_data[ltc_cols].head(5)
            print("\nLTC data sample:")
            print(ltc_sample)
        
        # Check if required columns exist in hospital data
        hosp_cols = ["STATE", "COUNTY", "BEDS", "LONGITUDE", "LATITUDE"]
        missing_hosp_cols = [col for col in hosp_cols if col not in hospital_data.columns]
        if missing_hosp_cols:
            print(f"WARNING: Missing columns in hospital data: {missing_hosp_cols}")
        else:
            print("All required columns present in hospital data")
            
            # Check for valid coordinate data
            hosp_sample = hospital_data[hosp_cols].head(5)
            print("\nHospital data sample:")
            print(hosp_sample)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing data loading...")
    test_data_loading() 