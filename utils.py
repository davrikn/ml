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
    for i in range(12):
        if not f'month_{i + 1}' in new_df:
            new_df[f'month_{i + 1}'] = False

    return new_df


def onehot_hours(df):
    new_df = df.copy()
    new_df['hour'] = df['date_forecast'].dt.hour
    # Perform one-hot encoding
    new_df = pd.get_dummies(new_df, columns=['hour', ])

    return new_df


def concat_observed_estimated(obs_df, est_df):
    new_df = pd.concat([obs_df, est_df], ignore_index=True)
    return new_df


def merge_train_target(train_df, target_df):
    merged_df = target_df.merge(train_df, left_on='time', right_on='date_forecast')
    return merged_df


def clean_and_plot_target_data(target_df, zero_threshold=150, constant_value_threshold=1):
    """
    Clean the target dataset by removing specified patterns and plot before and after cleaning.

    Parameters:
    - target_df: DataFrame containing the target values.
    - zero_threshold: The threshold for consecutive zero values to be removed.
    - constant_value_threshold: The threshold for consecutive constant non-zero values to be removed.

    Returns:
    - Cleaned target DataFrame.
    """

    # Working on a copy to avoid modifying the original DataFrame
    target_df = target_df.copy()

    target_df = target_df.dropna(subset=['pv_measurement'])

    # Identify sequences of zeros in the target
    zero_sequences = (target_df['pv_measurement'] == 0).astype(int).groupby(
        target_df['pv_measurement'].ne(0).cumsum()).cumsum()

    # Identify sequences of constant non-zero values in the target
    constant_sequences = (target_df['pv_measurement'].eq(target_df['pv_measurement'].shift()) & (
            target_df['pv_measurement'] != 0)).astype(int).groupby(
        target_df['pv_measurement'].ne(target_df['pv_measurement'].shift()).cumsum()).cumsum()

    # Mark rows to be removed
    remove_zeros = zero_sequences > zero_threshold
    remove_constants = constant_sequences > constant_value_threshold

    # Combine the conditions
    to_remove = remove_zeros | remove_constants

    total_data_points = len(target_df)
    removed_data_points = to_remove.sum()

    print(f"Total data points: {total_data_points}")
    print(f"Data points to be removed: {removed_data_points}")

    # Clean the target data
    cleaned_target = target_df[~to_remove].reset_index(drop=True)

    return cleaned_target


def preprocess_category(category: str):
    category = category.upper()

    target_df = pd.read_parquet(f'data/{category}/train_targets.parquet')
    estimated_df = pd.read_parquet(f'data/{category}/X_train_estimated.parquet')
    observed_df = pd.read_parquet(f'data/{category}/X_train_observed.parquet')
    test_df = pd.read_parquet(f'data/{category}/X_test_estimated.parquet')

    cleaned_target = clean_and_plot_target_data(target_df=target_df)

    # Observed
    observed_df['date_forecast'] = pd.to_datetime(observed_df['date_forecast'])
    observed_df.set_index('date_forecast', inplace=True)
    observed_df = observed_df.resample('H').mean()
    observed_df = observed_df.reset_index()

    observed_df = onehot_hours(observed_df)
    observed_df = onehot_months(observed_df)

    # Estimated
    estimated_df['date_forecast'] = pd.to_datetime(estimated_df['date_forecast'])
    estimated_df.set_index('date_forecast', inplace=True)
    estimated_df = estimated_df.resample('H').mean()
    estimated_df = estimated_df.reset_index()

    estimated_df = onehot_hours(estimated_df)
    estimated_df = onehot_months(estimated_df)

    # Test
    test_df['date_forecast'] = pd.to_datetime(test_df['date_forecast'])
    test_df.set_index('date_forecast', inplace=True)
    test_df = test_df.resample('H').mean()
    preprocessed_test = test_df.reset_index()

    preprocessed_test = onehot_hours(preprocessed_test)
    preprocessed_test = onehot_months(preprocessed_test)

    observed_target = merge_train_target(observed_df, cleaned_target)
    observed_target.fillna(0, inplace=True)

    estimated_target = merge_train_target(estimated_df, cleaned_target)
    estimated_target.fillna(0, inplace=True)

    return observed_target.reindex(sorted(observed_target.columns), axis=1), \
        estimated_target.reindex(sorted(estimated_target.columns), axis=1), \
        preprocessed_test.reindex(sorted(preprocessed_test.columns), axis=1)
