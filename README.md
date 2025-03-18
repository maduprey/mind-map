# Hospital Acquired Infections (HAI) Dashboard

This dashboard visualizes hospital acquired infections (HAI), specifically for Candida auris, using healthcare facility data from across the United States.

## Features

- Interactive map showing Long-term Care (LTC) facilities and hospitals across the US
- Visualization of the simulated infection network between facilities
- Interactive filtering by:
  - Minimum number of beds
  - Minimum connection strength between facilities

## Data Sources

- Long-term care facilities data from LTCFocus.org (2021)
- US hospital locations data

## Setup Instructions

1. Make sure you have Python 3.8+ installed

2. Clone this repository and navigate to the project directory

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the dashboard:
   ```
   streamlit run main.py
   ```

5. The dashboard should automatically open in your default web browser. If not, it will display a URL in the terminal that you can open.

## How to Use

1. **Facility Map**: Shows the locations of long-term care facilities (blue) and hospitals (red) across the US. The size of each point corresponds to the number of beds in the facility.

2. **Infection Network**: Displays a simulated social network of connections between hospitals and LTC facilities, representing potential infection transmission paths for Candida auris.

3. **Interactive Controls**:
   - Use the "Minimum Number of Beds" slider to filter facilities based on their capacity
   - Use the "Minimum Connection Strength" slider to show only the strongest connections in the infection network

## Notes

- The infection network is simulated for demonstration purposes using facility size and location data
- Connection strengths are not based on real infection data but are modeled based on facility characteristics and geographic proximity
- For performance reasons, the network visualization shows only a subset of the full network (focusing on the largest connected component) 