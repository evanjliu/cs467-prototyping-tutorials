import pandas as pd
import h3
import folium
from folium.plugins import HeatMap

# ------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------
HEX_RES_START = 10  # Start at high resolution
HEX_RES_MIN = 3  # Lowest resolution allowed
CALLS_PER_LEVEL = 2  # Every 2 extra calls reduce resolution

# Set the desired time range for visualization
# We can use this to dynamically adjust the resolution based on call volume
# Example we can use slider in front end to change the time range
TIME_START = "2023-01-01 00:00:00"
TIME_END = "2023-01-01 01:00:00"

# ------------------------------------------------------------
# Load and Process 911 Call Data
# ------------------------------------------------------------
file_path = "../../../CLT_data.csv"
calls_df = pd.read_csv(file_path, parse_dates=['Dispatched'])

# Convert timestamp column to datetime format
calls_df['Dispatched'] = pd.to_datetime(calls_df['Dispatched'], errors='coerce', format='%m/%d/%Y %H:%M')

# Remove rows with missing timestamps
calls_df.dropna(subset=['Dispatched'], inplace=True)

# Rename timestamp column for clarity
calls_df.rename(columns={'Dispatched': 'call_time'}, inplace=True)

# Sort calls by time
calls_df.sort_values('call_time', inplace=True)

# ------------------------------------------------------------
# Filter Data to Only Show Calls Within the Desired Time Range
# ------------------------------------------------------------
calls_df = calls_df[(calls_df['call_time'] >= TIME_START) & (calls_df['call_time'] <= TIME_END)]

# Assign each call to an H3 hexagon at initial high resolution
calls_df['hex_res'] = calls_df.apply(
    lambda row: h3.latlng_to_cell(row.Latitude, row.Longitude, HEX_RES_START),
    axis=1
    )

# ------------------------------------------------------------
# Group Calls by Initial Hexagons
# ------------------------------------------------------------
calls_grouped = calls_df.groupby('hex_res').size().reset_index(name="call_volume")


# ------------------------------------------------------------
# Function to Adjust H3 Resolution Based on Call Volume
# ------------------------------------------------------------
def dynamic_resolution(hex_id, call_count):
    resolution = HEX_RES_START
    while call_count >= CALLS_PER_LEVEL and resolution > HEX_RES_MIN:
        call_count -= CALLS_PER_LEVEL
        resolution -= 1
    return h3.cell_to_parent(hex_id, resolution)


# Apply dynamic resolution scaling
calls_grouped['adjusted_hex'] = calls_grouped.apply(lambda row: dynamic_resolution(row['hex_res'], row['call_volume']), axis=1)

# ------------------------------------------------------------
# Aggregate Calls for Adjusted Hexagons
# ------------------------------------------------------------
final_calls_df = calls_grouped.groupby('adjusted_hex')['call_volume'].sum().reset_index()

# ------------------------------------------------------------
# Generate Folium Heatmap
# ------------------------------------------------------------
# Get centroid locations for each hexagon
final_calls_df['lat_lon'] = final_calls_df['adjusted_hex'].apply(lambda h: h3.cell_to_latlng(h))

# Convert to list of lat/lon/weight for Folium HeatMap
heatmap_data = [[lat, lon, volume] for (lat, lon), volume in zip(final_calls_df['lat_lon'], final_calls_df['call_volume'])]

# Create a Folium map centered on Charlotte, NC
map = folium.Map(location=[35.2271, -80.8431], zoom_start=11)

# Add HeatMap layer
HeatMap(heatmap_data, radius=15, blur=10, max_zoom=1).add_to(map)

# Save and display map
map.save("911_call_dynamic_heatmap.html")
