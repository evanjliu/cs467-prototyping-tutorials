from tensorflow import keras
import pandas as pd
import h3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------
# Configuration & Hyperparameters
# ------------------------------------------------------------
HEX_RES_7 = 7  # Small hexagon resolution (for dense areas)
HEX_RES_5 = 5  # Larger hexagon resolution (for sparse areas)
CALL_THRESHOLD = 10  # Minimum number of calls to keep Res 7
EPOCHS = 20  # Training epochs
BATCH_SIZE = 32  # Batch size

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

# Assign each call to an H3 hexagon at resolution 7
def assign_h3_hex(row):
    return h3.latlng_to_cell(row.Latitude, row.Longitude, HEX_RES_7)

calls_df['hex_res_7'] = calls_df.apply(assign_h3_hex, axis=1)

# ------------------------------------------------------------
# Group Calls by Time & Location (Resolution 7)
# ------------------------------------------------------------
calls_grouped = calls_df.groupby(
    [pd.Grouper(key='call_time', freq='30min'), 'hex_res_7']
).size().reset_index(name="call_volume")

# Identify sparse hexagons (call count below threshold)
sparse_hexes = calls_grouped[calls_grouped['call_volume'] < CALL_THRESHOLD].copy()

# Convert sparse hexagons to parent hexagons at resolution 5
def convert_to_parent(hex_id):
    return h3.cell_to_parent(hex_id, HEX_RES_5)

sparse_hexes['hex_res_5'] = sparse_hexes['hex_res_7'].apply(convert_to_parent)

# ------------------------------------------------------------
# Aggregate Calls for Sparse Hexagons (Resolution 5)
# ------------------------------------------------------------
sparse_hex_agg = sparse_hexes.groupby(['call_time', 'hex_res_5'])['call_volume'].sum().reset_index()

# Remove sparse hexagons from the original data
high_density_hexes = calls_grouped[calls_grouped['call_volume'] >= CALL_THRESHOLD].copy()

# ------------------------------------------------------------
# Merge High-Density & Low-Density Data
# ------------------------------------------------------------
# Rename columns for consistency
sparse_hex_agg.rename(columns={'hex_res_5': 'hex_id'}, inplace=True)
high_density_hexes.rename(columns={'hex_res_7': 'hex_id'}, inplace=True)

# Combine both datasets into one
final_calls_df = pd.concat([high_density_hexes, sparse_hex_agg], ignore_index=True)

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------
# Extract time-based features
final_calls_df['hour_of_day'] = final_calls_df['call_time'].dt.hour
final_calls_df['day_of_week'] = final_calls_df['call_time'].dt.dayofweek

# Convert hex IDs into numerical category codes
final_calls_df['hex_id_encoded'] = final_calls_df['hex_id'].astype('category').cat.codes

# ------------------------------------------------------------
# Prepare Data for LSTM Model
# ------------------------------------------------------------
# Define input features (drop non-numeric columns)
feature_inputs = final_calls_df.drop(columns=['call_time', 'call_volume', 'hex_id']).values

# Define target variable (number of calls)
call_targets = final_calls_df['call_volume'].values

# Reshape for LSTM input format (samples, time steps, features)
feature_inputs = feature_inputs.reshape((feature_inputs.shape[0], 1, feature_inputs.shape[1]))

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
# 80% training, 20% testing
split_index = int(len(feature_inputs) * 0.8)

X_train_set = feature_inputs[:split_index]
X_test_set = feature_inputs[split_index:]

Y_train_set = call_targets[:split_index]
Y_test_set = call_targets[split_index:]

print(f"Training Set Size: {X_train_set.shape}, Test Set Size: {X_test_set.shape}")

# ------------------------------------------------------------
# Build & Train LSTM Model
# ------------------------------------------------------------
lstm_model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(1, X_train_set.shape[2])),
    keras.layers.LSTM(32, activation='relu'),
    keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

print("Model Summary:")
print(lstm_model.summary())

# Train the model
training_results = lstm_model.fit(
    X_train_set, Y_train_set,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_set, Y_test_set),
    verbose=1
)

# ------------------------------------------------------------
# Predictions & Visualization
# ------------------------------------------------------------
predicted_calls = lstm_model.predict(X_test_set)

# Number of data points to show (adjust as needed)
time_window = 500  # Show only the last 500 time steps

# Get the last `time_window` entries
date_range = final_calls_df['call_time'].iloc[-time_window:]
y_actual = Y_test_set[-time_window:]
y_predicted = predicted_calls[-time_window:]

# Convert timestamps to matplotlib date format
dates = mdates.date2num(date_range)

# ------------------------------------------------------------
# Plotting with Date Format
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(dates, y_actual, label="Actual Calls", color="blue", linewidth=1)
plt.plot(dates, y_predicted, label="Predicted Calls", linestyle='dashed', color="red", linewidth=1)

# Format x-axis to show dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xlabel("Time")
plt.ylabel("911 Call Volume")
plt.title("911 Call Predictions with Adaptive H3 Resolution")

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)

plt.show()
