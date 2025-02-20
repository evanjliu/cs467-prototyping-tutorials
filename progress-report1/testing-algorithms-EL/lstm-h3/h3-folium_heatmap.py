import pandas as pd
import h3
import folium
import json
from folium.plugins import HeatMap

# ------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------
HEX_RES_START = 10  # Start at high resolution (smaller hexagons)
HEX_RES_MIN = 4  # Lowest resolution allowed (larger hexagons)
CALLS_THRESHOLD_MAX = 12 # Minimum calls needed to keep resolution at HEX_RES_START
CALLS_PER_LEVEL = CALLS_THRESHOLD_MAX / abs(HEX_RES_START - HEX_RES_MIN)  # Every X extra calls reduce resolution


# ------------------------------------------------------------
# Load Predictions from JSON File
# ------------------------------------------------------------
def load_lstm_predictions(prediction_file):
    with open(prediction_file, "r") as json_file:
        lstm_predictions = json.load(json_file)

    return pd.DataFrame(lstm_predictions)


# ------------------------------------------------------------
# Function to filter predictions by time parameters
# ------------------------------------------------------------
def filter_predictions_by_time(predictions_dataframe, time_start, time_end):
    filtered_df = predictions_dataframe[
        (predictions_dataframe["call_time"] >= time_start) &
        (predictions_dataframe["call_time"] <= time_end)
    ]

    return filtered_df


# ------------------------------------------------------------
# Function to Adjust H3 Resolution Based on Call Volume
# ------------------------------------------------------------
def dynamic_resolution(hex_id, call_count):
    resolution = HEX_RES_START
    while call_count < CALLS_THRESHOLD_MAX and resolution > HEX_RES_MIN:
        call_count += CALLS_PER_LEVEL
        resolution -= 1
    return h3.cell_to_parent(hex_id, resolution)

# ------------------------------------------------------------
# Function to create fixed resolution heatmap
# 
# Accepts a prediction file, time range, and output file
# Expects that the prediction file contains hexagons at varying reolutions
# Meaning that the model trained off of dynamically changing resolutions.
# ------------------------------------------------------------
def create_heatmap_fixed(prediction_file, time_start="2023-01-01 00:00:00", time_end="2023-01-01 01:00:00", output_map="./heatmaps_predictions/heatmap.html"):
    predictions_dataframe = load_lstm_predictions(prediction_file)
    predictions_dataframe = filter_predictions_by_time(
        predictions_dataframe,
        time_start,
        time_end
        )

    # Convert hex region ID back to H3 index (from JSON resolution)
    predictions_dataframe["h3_hex_id"] = predictions_dataframe["hex_region_id"]

    # Get center locations for each Hexagon (centroid)
    predictions_dataframe["hex_center"] = [h3.cell_to_latlng(hex) for hex in predictions_dataframe["h3_hex_id"]]

    # Convert to list for Folium heatmap
    heatmap_data = [[lat, lon, volume] for (lat, lon), volume in zip(
        predictions_dataframe["hex_center"], predictions_dataframe["predicted_call_volume"]
    )]

    # Generate Folium heatmap
    # Charlotte, NC coordinates
    map_center = [35.2271, -80.8431]
    map = folium.Map(location=map_center, zoom_start=11)

    # Add heatmap to the map
    # Save and display the map
    heat_layer = HeatMap(heatmap_data, radius=15, blur=10, max_zoom=1)
    map.add_child(heat_layer)
    map.save(output_map)


# ------------------------------------------------------------
# Function to create dynamic resolution scaling heatmap
# 
# Accepts a prediction file, start and end times, and an output file path
# Expects that all resolutions of provided hexagons are the same.
# Meaning that the model trained off of a single resolution.
# ------------------------------------------------------------
def create_heatmap_dynamic(prediction_file, time_start="2023-01-01 00:00:00", time_end="2023-01-01 01:00:00", output_map="./heatmaps_predictions/heatmap.html"):
    predictions_dataframe = load_lstm_predictions(prediction_file)
    predictions_dataframe = filter_predictions_by_time(
        predictions_dataframe,
        time_start,
        time_end
    )

    # Apply dynamic resolution scaling
    predictions_dataframe["adjusted_hex"] = predictions_dataframe.apply(
        lambda row: dynamic_resolution(row["hex_region_id"], row["predicted_call_volume"]),
        axis=1
    )

    # Group by adjusted hexagon and sum call volumes
    grouped_calls = predictions_dataframe.groupby("adjusted_hex")

    # Add and agregate call volumes
    call_sums = grouped_calls["predicted_call_volume"].sum()

    # Convert back to DataFrame and reset index
    final_calls_df = call_sums.reset_index()

    # Get centroid locations for each parent adjusted hexagon
    final_calls_df["lat_lon"] = [h3.cell_to_latlng(hex) for hex in final_calls_df["adjusted_hex"]]

    # Convert to list for Folium heatmap
    heatmap_data = [[lat, lon, volume] for (lat, lon), volume in zip(final_calls_df["lat_lon"], final_calls_df["predicted_call_volume"])]

    # Generate Folium heatmap
    #  Charlotte, NC coordinates
    map_center = [35.2271, -80.8431]
    map = folium.Map(location=map_center, zoom_start=11)

    # Add heatmap to the map
    # Save and display the map
    heat_layer = HeatMap(heatmap_data, radius=15, blur=10, max_zoom=1)
    map.add_child(heat_layer)
    map.save(output_map)


if __name__ == "__main__":
    # Example Usage
    create_heatmap_fixed(
        prediction_file="./heatmaps_predictions/lstm_predictions.json",
        time_start="2023-01-01 00:00:00",
        time_end="2023-01-01 01:00:00",
        output_map="./heatmaps_predictions/heatmap_fixed.html"
    )
    