from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import utils
import numpy as np


def do_prediction(location, limit, name, percentage):
    x_train, tuning_data, x_test = utils.preprocess_category_estimated_observed(location)
    x_train.drop(["time", 'date_forecast'], axis=1, inplace=True)
    tuning_data.drop(["time", 'date_forecast'], axis=1, inplace=True)
    x_test_date_forecast = x_test['date_forecast']
    x_test.drop(['date_forecast'], axis=1, inplace=True)

    x_test.fillna(0, inplace=True)

    label = 'pv_measurement'
    train_data = TabularDataset(x_train)

    precentage_tuning = percentage / 100

    tuning_data = TabularDataset(tuning_data)
    thirty_percent_index = int(len(tuning_data) * precentage_tuning)
    tuning_data = tuning_data.iloc[:thirty_percent_index]

    test_data = TabularDataset(x_test)

    predictor = TabularPredictor(label=label,
                                 path="AutoGluonTesting",
                                 eval_metric='mean_absolute_error')

    num_trials = 60  # try at most 20 different hyperparameter configurations for each type of model
    search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

    hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        'num_trials': num_trials,
        'scheduler': 'local',
        'searcher': search_strategy,
    }

    predictor.fit(train_data,
                  time_limit=limit,
                  tuning_data=tuning_data,
                  hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, )

    y_pred = predictor.predict(test_data)

    print(y_pred)

    preds = pd.DataFrame()
    preds['date_forecast'] = x_test_date_forecast
    preds['predicted'] = np.asarray(y_pred)
    preds.to_csv(name + "_" + str(percentage) + '_' + location + '.csv')
    print('Saved this file: ' + name + '_' + str(percentage) + '_' + location + '.csv')


if __name__ == '__main__':
    time_limit = 60 * 60
    percentage = 10 + 2 * 10
    name = str(1) + "_tuning_60HPO_"
    print('Starting run with percentage tuning= ' + str(percentage))
    do_prediction('A', time_limit, name, percentage)
    do_prediction('B', time_limit, name, percentage)
    do_prediction('C', time_limit, name, percentage)
    print('Done with run with percentage tuning= ' + str(percentage))
