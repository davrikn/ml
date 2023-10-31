import pandas as pd



def build_pandas(location: str):
    train_targets = pd.read_parquet(f'./data/{location}/train_targets.parquet')
    x_test_estimated = pd.read_parquet(f'./data/{location}/X_test_estimated.parquet')
    x_train_estimated = pd.read_parquet(f'./data/{location}/X_train_estimated.parquet')
    x_train_observed = pd.read_parquet(f'./data/{location}/X_train_observed.parquet')

    train_targets.rename(columns={'time': 'date_forecast'}, inplace=True)

    overlap = x_train_estimated['date_forecast'].isin(x_train_observed['date_forecast']).any()
    if overlap:
        print("Overlap found")

    x_train = pd.concat([x_train_estimated, x_train_observed], join='outer').drop('date_calc', axis=1)
    #x_train = x_train.merge(train_targets[['date_forecast', 'pv_measurement']], on='date_forecast', how='inner')
    x_train.fillna(0, inplace=True)
    x_train['date_forecast'] = pd.to_datetime(x_train['date_forecast'])
    x_train.sort_values(by='date_forecast', inplace=True)

    x_test_estimated.fillna(0, inplace=True)

    x_train = pd.merge(x_train, train_targets, on='date_forecast', how='inner')

    return x_train, x_test_estimated