import pandas as pd


def prepare_dataframe(file_path, time_step="h"):
    """
    Prepares dataframe for analysis by filtering to only show rows where
    CauseCategory == EMS and converting the 'Dispatched' column to a datetime
    object.

    Input:
        file_path (str): Path to the CSV file containing the data.
        time_step (str): Resampling time step ("h" for hourly, "d" for daily).

    Returns:
        Resampled dataframe with call_count column.
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
