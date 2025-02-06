import itertools
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and prepare data. Change time_step to match hourly/daily
from data_process import prepare_dataframe
file_path = "../../../CLT_data.csv"
time_step = "h"
df = prepare_dataframe(file_path, time_step)

# Prepare data without 2021 data (Unused for now)
# df_no_2021 = df[~(df.index.year == 2021)].copy()

# Define hyperparameter options
trend_options = ['add', 'mul', None]
seasonal_options = ['add', 'mul', None]

# Option to also test including and excluding 2021 data because of the pandemic (Unused for now)
# include_2021 = [True, False]

# Seasonal period options, one is daily and the other is hourly
if time_step == "h":
    # Hourly (Daily/Weekly)
    seasonal_periods_options = [24, 168]
else:
    # Daily (Weekly/Monthly)
    seasonal_periods_options = [7, 30]

# Generate all possible hyperparameter combinations into a list of tuples
param_grid = list(itertools.product(
    trend_options,
    seasonal_options,
    seasonal_periods_options
    ))

# Print total number of hyperparameter combinations
print(f"Total hyperparameter combinations: {len(param_grid)}")
best_score = float("inf")
best_params = None
results = []

# Loop through each hyperparameter combination and display MAE and RMSE
for trend, seasonal, seasonal_period in param_grid:
    try:
        # Fit Holt-Winters model
        hw_model = ExponentialSmoothing(
            df["call_count"],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_period
            )

        # Get predictions
        df["forecast"] = hw_model.fit().fittedvalues

        # Calculate MAE & RMSE
        mae = mean_absolute_error(df["call_count"], df["forecast"])
        rmse = np.sqrt(mean_squared_error(df["call_count"], df["forecast"]))

        # Store results
        results.append((trend, seasonal, seasonal_period, mae, rmse))

        # Track best model
        if mae < best_score:
            best_score = mae
            best_params = (trend, seasonal, seasonal_period)

        print(f"Evaluated: trend={trend}, seasonal={seasonal}, period={seasonal_period} → MAE={mae:.2f}, RMSE={rmse:.2f}")

    except Exception as e:
        print(f"Error with parameters: trend={trend}, seasonal={seasonal}, period={seasonal_period} - {e}")

print("\nBest Model Parameters:")
print(f"Trend: {best_params[0]}")
print(f"Seasonal: {best_params[1]}")
print(f"Seasonal Periods: {best_params[2]}")
print(f"Best MAE: {best_score:.2f}")
