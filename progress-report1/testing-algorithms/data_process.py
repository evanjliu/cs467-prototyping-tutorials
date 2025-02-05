import pandas as pd

def prepare_dataframe_hourly(file_path):
    """
    Prepares dataframe for analysis by filtering to only show rows where CauseCategory == EMS
    and converting the 'Dispatched' column to a datetime object.
    
    Returns:
        Resampled dataframe with call_count column.
    """
    df = pd.read_csv(file_path)
    
    # Code from Ryan to prepare data to datetime and correct formatting
    df.drop(columns=['Unique ID', 'Nature Code','State Plane Feet X', 'State Plane Feet Y', 'Shift', 'Battalion', 'Division', 'DispatchNature'], inplace=True)
    df['e'] = 1
    filtered_df = df[df['CauseCategory'] == 'EMS']
    filtered_df['Dispatched'] = pd.to_datetime(filtered_df['Dispatched'], format='%m/%d/%Y %H:%M')
    
    # Set datetime as index then resample to hourly/daily intervals
    # Change "h" to "d" to resample to daily intervals and vice versa.
    filtered_df.set_index('Dispatched', inplace=True)
    df_resampled = filtered_df.resample("h").size().to_frame(name="call_count")
    
    
    return df_resampled

def prepare_dataframe_daily(file_path):
    """
    Prepares dataframe for analysis by filtering to only show rows where CauseCategory == EMS
    and converting the 'Dispatched' column to a datetime object.
    
    Returns:
        Resampled dataframe with call_count column.
    """
    df = pd.read_csv(file_path)
    
    # Code from Ryan to prepare data to datetime and correct formatting
    df.drop(columns=['Unique ID', 'Nature Code','State Plane Feet X', 'State Plane Feet Y', 'Shift', 'Battalion', 'Division', 'DispatchNature'], inplace=True)
    df['e'] = 1
    filtered_df = df[df['CauseCategory'] == 'EMS']
    filtered_df['Dispatched'] = pd.to_datetime(filtered_df['Dispatched'], format='%m/%d/%Y %H:%M')
    
    # Set datetime as index then resample to hourly/daily intervals
    # Change "h" to "d" to resample to daily intervals and vice versa.
    filtered_df.set_index('Dispatched', inplace=True)
    df_resampled = filtered_df.resample("d").size().to_frame(name="call_count")
    
    
    return df_resampled