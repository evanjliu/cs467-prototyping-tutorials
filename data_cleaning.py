import pandas as pd
import matplotlib.pyplot as plt

# this reads the csv file into a dataframe
df = pd.read_csv("CLT_FY18-24_Categorized.csv")

# shows info about each column
df.info()

# displays the first 5 rows
print(df.head())

# displays column names
print(df.columns)

# shows descriptive statistics
print(df.describe())

# shows missing values count
print(df.isnull().sum())

# columns we want to convert to datetime
time_cols = ["Dispatched", "FirstResponding", "FirstArrival", "FullComplement"]
for col in time_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# add columns for hour and day of week
if "Dispatched" in df.columns:
    df["Dispatched_Hour"] = df["Dispatched"].dt.hour
    df["Dispatched_DayOfWeek"] = df["Dispatched"].dt.dayofweek

# selected categorical columns to check value counts
cat_cols = ["Nature Code", "DispatchNature", "CauseCategory"]
for col in cat_cols:
    if col in df.columns:
        print(df[col].value_counts(dropna=False))

# simple bar chart for calls by causecategory
if "CauseCategory" in df.columns:
    df["CauseCategory"].value_counts().plot(
        kind="bar", title="Calls by CauseCategory")
    plt.show()

# check min and max latitude/longitude
if "Latitude" in df.columns and "Longitude" in df.columns:
    print(df["Latitude"].min(), df["Latitude"].max())
    print(df["Longitude"].min(), df["Longitude"].max())

# display a random sample of rows
print(df.sample(5))
