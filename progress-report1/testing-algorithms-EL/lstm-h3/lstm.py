import pandas as pd
import h3
import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------
MAX_RESOLUTION = 10  # Starting resolution for H3 hexagons
MIN_RESOLUTION = 4  # Minimum allowed resolution
CALLS_AT_MAX_RESOLUTION = 12  # Minimum calls needed to keep resolution at MAX_RESOLUTION
CALLS_PER_RESOLUTION_STEP = CALLS_AT_MAX_RESOLUTION / (MAX_RESOLUTION - MIN_RESOLUTION)  # Calls needed to increase resolution

TIME_WINDOW = "30min"  # Aggregate calls in 30-minute intervals

SEQUENCE_LENGTH = 10  # Time steps for LSTM input
LSTM_NEURONS = 64  # Number of neurons in LSTM layer
LEARNING_RATE = 0.001  # Learning rate for optimizer
NUM_EPOCHS = 5  # Number of training epochs
BATCH_SIZE = 64  # Training batch size

# ------------------------------------------------------------
# Function to Adjust H3 Resolution Based on Call Volume
# TODO
# ------------------------------------------------------------
def adjust_hex_resolution(hex_id, call_count):
    resolution = MAX_RESOLUTION  # Start at the highest resolution (smallest hexagons)

    if call_count == 0:
        return h3.cell_to_parent(hex_id, MIN_RESOLUTION)

    # Merge hexagons when call count is too low merge into parent hexagon
    while call_count < CALLS_AT_MAX_RESOLUTION and resolution > MIN_RESOLUTION:
        call_count += CALLS_PER_RESOLUTION_STEP  
        resolution -= 1

    return h3.cell_to_parent(hex_id, resolution)


def lat_lng_to_h3(row):
    return h3.latlng_to_cell(row["Latitude"], row["Longitude"], MAX_RESOLUTION)


def adjust_hex(row):
    return adjust_hex_resolution(row["hex_id"], row["call_volume"])


# ------------------------------------------------------------
# Function to Preprocess 911 Call Data for LSTM
# ------------------------------------------------------------
def preprocess_data(file_path):
    # Load CSV file
    raw_data = pd.read_csv(file_path, parse_dates=["Dispatched"])

    # Filter only EMS calls
    raw_data = raw_data[raw_data["CauseCategory"] == "EMS"]

    # Convert lat/lon to H3 hexagons
    raw_data["hex_id"] = raw_data.apply(lat_lng_to_h3, axis=1)

    # Round timestamps to the nearest time window
    raw_data["time_window"] = raw_data["Dispatched"].dt.floor("min")

    # Aggregate call volume per time window and hexagon
    aggregated_data = raw_data.groupby(["time_window", "hex_id"]).size()
    aggregated_data = aggregated_data.reset_index(name="call_volume")

    # Apply dynamic hex resolution adjustment
    aggregated_data["adjusted_hex"] = aggregated_data.apply(adjust_hex, axis=1)

    # Aggregate again after adjusting resolution
    final_data = aggregated_data.groupby(["time_window", "adjusted_hex"], as_index=False)["call_volume"].sum()

    return final_data

# ------------------------------------------------------------
# Function to Prepare Data for LSTM Training
# ------------------------------------------------------------
def prepare_lstm_data(call_data):
    scaler = MinMaxScaler(feature_range=(0, 1))

    call_volumes = call_data["call_volume"].values
    call_volumes = call_volumes.reshape(-1, 1)

    normalized_data = scaler.fit_transform(call_volumes)

    return normalized_data, scaler

# ------------------------------------------------------------
# Function to Convert Data into LSTM Sequences
# ------------------------------------------------------------
def create_sequences(data, sequence_length):
    x_sequences = []
    y_sequences = []

    for i in range(len(data) - sequence_length):
        x_seq = data[i:i + sequence_length]
        y_seq = data[i + sequence_length]

        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    x_sequences = np.array(x_sequences)
    y_sequences = np.array(y_sequences)

    return x_sequences, y_sequences

# ------------------------------------------------------------
# LSTM Model Definition
# ------------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(LSTM(units=LSTM_NEURONS, return_sequences=True))
    model.add(LSTM(units=LSTM_NEURONS, return_sequences=True))
    model.add(LSTM(units=LSTM_NEURONS))

    model.add(Dense(units=1))  # Single value output for predicted call volume

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss="mse")

    return model

# ------------------------------------------------------------
# Function to Train LSTM Model
# ------------------------------------------------------------
def train_lstm_model(train_data, sequence_length=SEQUENCE_LENGTH, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    x_train, y_train = create_sequences(train_data, sequence_length)

    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    model = build_lstm_model(input_shape=(sequence_length, 1))

    model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs, 
        batch_size=batch_size, 
        verbose=1
    )

    return model

# ------------------------------------------------------------
# Function to Generate Predictions with LSTM
# ------------------------------------------------------------
def generate_lstm_predictions(model, test_data, sequence_length=SEQUENCE_LENGTH):
    x_test, _ = create_sequences(test_data, sequence_length)

    x_test = np.expand_dims(x_test, axis=-1)

    predictions = model.predict(x_test)

    predictions = predictions.flatten()

    return predictions

# ------------------------------------------------------------
# Function to Save LSTM Predictions as JSON
# ------------------------------------------------------------
def save_lstm_predictions_to_json(predictions, hex_ids, timestamps, output_file="lstm_predictions.json"):
    prediction_data = []

    for i in range(len(predictions)):
        entry = {
            "hex_region_id": hex_ids[i],
            "predicted_call_volume": float(predictions[i]),
            "call_time": timestamps[i]
        }

        prediction_data.append(entry)

    with open(output_file, "w") as json_file:
        json.dump(prediction_data, json_file)

    print("Saved predictions")

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "../../../../CLT_data.csv" 
    processed_data = preprocess_data(file_path)

    # Prepare data for LSTM
    lstm_ready_data, scaler = prepare_lstm_data(processed_data)

    # Train LSTM model
    trained_model = train_lstm_model(lstm_ready_data)

    # Generate predictions
    predictions = generate_lstm_predictions(trained_model, lstm_ready_data)

    # Convert predictions back to original scale
    predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1))
    predictions_original_scale = predictions_original_scale.flatten()

    # Generate timestamps for predictions
    min_time = processed_data["time_window"].min()
    num_predictions = len(predictions)

    timestamps = pd.date_range(
        start=min_time, 
        periods=num_predictions, 
        freq="30min"
    )

    timestamps = timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist()

    # Save predictions to JSON
    save_lstm_predictions_to_json(
        predictions_original_scale, 
        processed_data["adjusted_hex"].tolist(), 
        timestamps
    )
