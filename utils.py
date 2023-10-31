import pandas as pd

def interpolate_quarter_hourly(df):
    """
    Interpolate the DataFrame to include quarter-hourly data.

    Parameters:
    - df: The input DataFrame with columns 'time' and 'pv_measurement'.

    Returns:
    - A new DataFrame with interpolated quarter-hourly data.
    """
    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()
    
    # Ensure the 'time' column is in datetime format
    df_copy['time'] = pd.to_datetime(df_copy['time'])
    
    # Set 'time' as the index and sort the DataFrame by it
    df_copy.set_index('time', inplace=True)
    df_copy.sort_index(inplace=True)

    # Resample the DataFrame to every 15 minutes and use linear interpolation for missing values
    df_resampled = df_copy.resample('15T').asfreq()
    df_interpolated = df_resampled.interpolate(method='time')

    # Fill any remaining missing values with a method of your choice (forward fill, backward fill, etc.)
    df_interpolated.fillna(method='ffill', inplace=True)

    # Reset the index to convert 'time' back to a column
    df_interpolated.reset_index(inplace=True)

    return df_interpolated

def onehot_months(df):
    new_df = df.copy()
    new_df['month'] = df['date_forecast'].dt.month
    # Perform one-hot encoding
    new_df = pd.get_dummies(new_df, columns=['month', ])
    
    return new_df

def onehot_hours(df):
    new_df = df.copy()
    new_df['hour'] = df['date_forecast'].dt.hour
    # Perform one-hot encoding
    new_df = pd.get_dummies(new_df, columns=['hour', ])

    return new_df

def concat_observed_estimated(obs_df, est_df):
    new_df = pd.concat([obs_df,est_df], ignore_index=True)
    return new_df

def merge_train_target(train_df, target_df):
    merged_df = target_df.merge(train_df, left_on='time', right_on='date_forecast')
    return merged_df

def clean_data(target_df, estimated_df, observed_df, zero_threshold=200, constant_value_threshold=1):
    """
    Clean the datasets by removing specified patterns, using the "time" column as the ID.

    Parameters:
    - target_df: DataFrame containing the target values.
    - estimated_df, observed_df, test_df: DataFrames to be cleaned based on the target DataFrame.
    - zero_threshold: The threshold for consecutive zero values to be removed.
    - constant_value_threshold: The threshold for consecutive constant non-zero values to be removed.

    Returns:
    - Cleaned target, estimated, observed, and test DataFrames.
    """
    
    # Identify sequences of zeros in the target
    zero_sequences = (target_df['pv_measurement'] == 0).astype(int).groupby(target_df['pv_measurement'].ne(0).cumsum()).cumsum()

    # Identify sequences of constant non-zero values in the target
    constant_sequences = target_df['pv_measurement'].ne(target_df['pv_measurement'].shift()).groupby(target_df['pv_measurement']).cumsum()

    # Mark rows to be removed
    remove_zeros = zero_sequences > zero_threshold
    remove_constants = target_df['pv_measurement'].duplicated(keep=False) & (target_df['pv_measurement'] != 0)

    # Combine the conditions
    to_remove = remove_zeros | remove_constants

    # Extract the "time" IDs of the rows to be removed
    time_ids_to_remove = target_df.loc[to_remove, 'time']

    # Remove marked rows from all DataFrames based on the "time" IDs
    cleaned_target = target_df[~target_df['time'].isin(time_ids_to_remove)].reset_index(drop=True)
    cleaned_estimated = estimated_df[~estimated_df['time'].isin(time_ids_to_remove)].reset_index(drop=True)
    cleaned_observed = observed_df[~observed_df['time'].isin(time_ids_to_remove)].reset_index(drop=True)

    return cleaned_target, cleaned_estimated, cleaned_observed

def preprocess_category(category: str):
    category = category.upper()

    target_df = pd.read_parquet(f'data/{category}/train_targets.parquet')
    estimated_df = pd.read_parquet(f'data/{category}/X_train_estimated.parquet')
    observed_df = pd.read_parquet(f'data/{category}/X_train_observed.parquet')

    cleaned_target, cleaned_estimated, cleaned_observed = clean_data(target_df=target_df, estimated_df=estimated_df,
                                                                     observed_df=observed_df)

    preprocessed_df = concat_observed_estimated(cleaned_observed, cleaned_estimated)
    preprocessed_df = onehot_hours(preprocessed_df)
    preprocessed_df = onehot_months(preprocessed_df)
    
    target_interpolated = interpolate_quarter_hourly(cleaned_target)

    merged_df = merge_train_target(preprocessed_df, target_interpolated)

    return merged_df

