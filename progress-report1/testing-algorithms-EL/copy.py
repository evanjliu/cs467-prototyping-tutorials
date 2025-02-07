import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


def prepare_dataframe(file_path, time_step="h"):
    """
    Prepare the dataframe for the by resampling the data to the given time_step.
    The data can be split into hourly ("h") and daily ("d") intervals.
    """
    df = pd.read_csv(file_path)

    # Code from Ryan to prepare data to datetime and prepare correct formatting
    df.drop(columns=['Unique ID', 'Nature Code', 'State Plane Feet X',
                     'State Plane Feet Y', 'Shift', 'Battalion', 'Division',
                     'DispatchNature'], inplace=True)
    df['e'] = 1
    filtered_df = df[df['CauseCategory'] == 'EMS'].copy()
    filtered_df['Dispatched'] = pd.to_datetime(
        filtered_df['Dispatched'],
        format='%m/%d/%Y %H:%M'
        )

    # Set datetime as index then resample in intervals of time_step
    filtered_df.set_index('Dispatched', inplace=True)
    df_resampled = filtered_df.resample(time_step).size().to_frame(
        name="call_count"
        )

    return df_resampled


def fit_es_model(train_data, trend="mul", seasonal="add", seasonal_periods=168):
    """
    Fit an Exponential Smoothing model to the given training data.
    The model can be configured with the following parameters:
    - trend: "add" or "mul" or None
    - seasonal: "add" or "mul" or None
    - seasonal_periods: int
    """
    es_model = ExponentialSmoothing(
        train_data["call_count"],
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
        ).fit()
    return es_model


def fit_lstm_model(scaled_training_data, sequence_length=24, epochs=10, batch_size=32):
    """
    Fit an LSTM model to the scaled training data. 
    This model can be configured with the following parameters:
    - sequence_length: int
    - epochs: int
    - batch_size: int
    """
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Create sequences using TimeseriesGenerator
    # The input data should be the "scaled_call_count" column
    # The target data should be the same column, shifted by sequence_length
    train_values = scaled_training_data["scaled_call_count"].values
    generator = TimeseriesGenerator(
        train_values,  
        train_values, 
        length=sequence_length,
        batch_size=batch_size
    )

    model.fit(generator, epochs=epochs, verbose=1)
    return model


if __name__ == "__main__":
    # Varables to update the LSTM model
    sequence_length = 24
    epochs = 10
    batch_size = 32
    file_path = "../../../CLT_data.csv"   
    time_step = "h"

    # Prepare the dataframe calling prepare_dataframe function
    df = prepare_dataframe(file_path, time_step)

    # Split the data into training and testing sets
    split_index = int(len(df) * 0.8)  
    training_data = df.iloc[:split_index].copy()
    testing_data = df.iloc[split_index:].copy()

    # Scale the training data using MinMaxScaler for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_data["scaled_call_count"] = scaler.fit_transform(training_data[["call_count"]])
    testing_data["scaled_call_count"] = scaler.transform(testing_data[["call_count"]])

    # Fit the ES model
    es_model = fit_es_model(training_data, trend="add", seasonal="add", seasonal_periods=168)

    # Make predictions using the ES model
    es_forcast = es_model.forecast(steps=len(testing_data))

    # Store the predictions in the testing_data DataFrame
    testing_data["es_forecast"] = es_forcast

    # Fit the LSTM model
    lstm_model = fit_lstm_model(training_data, sequence_length=sequence_length, epochs=epochs, batch_size=batch_size)

    # We create a new generator for the testing data to make predictions
    test_values = testing_data["scaled_call_count"].values

    # Test values
    test_generator = TimeseriesGenerator(
        test_values,
        test_values,
        length=sequence_length,
        batch_size=1
    )

    # Make predictions using the LSTM model
    lstm_forecast = lstm_model.predict(test_generator)

    # Before storing the predictions in the testing_data dataframe, we need to inverse the scaling
    lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()

    # The testing_data dataframe has an extra sequence_length rows at the beginning so we need
    # to remove it by slcing the original testing_data dataframe.
    testing_data = testing_data.iloc[sequence_length:]

    # Store predictions in the testing_data dataframe
    testing_data["lstm_forecast"] = lstm_forecast

    # Combine the ES and LSTM forecasts into a forecast
    testing_data["hybrid_forecast"] = 0.2 * testing_data["es_forecast"] + 0.8 * testing_data["lstm_forecast"]

    # Calculate the MAE for both models
    es_mae = mean_absolute_error(testing_data["call_count"], testing_data["es_forecast"])
    lstm_mae = mean_absolute_error(testing_data["call_count"], testing_data["lstm_forecast"])
    hybrid_mae = mean_absolute_error(testing_data["call_count"], testing_data["hybrid_forecast"])
    print(f"ES MAE: {es_mae}\nLSTM MAE: {lstm_mae}\nHybrid MAE: {hybrid_mae}")

    plt.figure(figsize=(12, 6))
    plt.plot(testing_data.index, testing_data["call_count"], label="Actual Calls")
    plt.plot(testing_data.index, testing_data["es_forecast"], label="ES Forecast", linestyle="solid", color="red")
    plt.plot(testing_data.index, testing_data["lstm_forecast"], label="LSTM Forecast", linestyle="solid", color="blue")
    plt.plot(testing_data.index, testing_data["hybrid_forecast"], label="Hybrid Forecast", linestyle="solid", color="purple")
    plt.legend()
    plt.title("Hybrid Model Forecast vs Actual Calls")
    plt.xlabel("Time")
    plt.ylabel("Call Volume")
    plt.show()
