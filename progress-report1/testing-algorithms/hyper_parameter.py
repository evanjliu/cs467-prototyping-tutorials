import itertools
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and prepare data
from data_process import prepare_dataframe_daily
print("Current working directory:", os.getcwd())
file_path = os.path.join(os.getcwd(), "data", "CLT_data.csv")
df = prepare_dataframe_daily(file_path)

# Define hyperparameter options
trend_options = ['add', 'mul', None]  
seasonal_options = ['add', 'mul', None]  
seasonal_periods_options = [7, 30] 

# Generate all possible hyperparameter combinations
param_grid = list(itertools.product(trend_options, seasonal_options, seasonal_periods_options))

print(f"Total hyperparameter combinations: {len(param_grid)}")
best_score = float("inf")
best_params = None
results = []

for trend, seasonal, seasonal_period in param_grid:
    try:
        # Fit Holt-Winters model
        hw_model = ExponentialSmoothing(df["call_count"], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period)
        hw_fitted = hw_model.fit()
        
        # Get predictions
        df["HW_Forecast"] = hw_fitted.fittedvalues

        # Calculate MAE & RMSE
        mae = mean_absolute_error(df["call_count"], df["HW_Forecast"])
        rmse = np.sqrt(mean_squared_error(df["call_count"], df["HW_Forecast"]))

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
