from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import utils
import numpy as np


def do_prediction(location, limit, name):
    x_train, tuning_data, x_test = utils.preprocess_category_estimated_observed(location)
    x_train.drop(["time"], axis=1, inplace=True)
    tuning_data.drop(["time"], axis=1, inplace=True)

    x_train['date_forecast'] = pd.to_datetime(x_train['date_forecast'])
    tuning_data['date_forecast'] = pd.to_datetime(tuning_data['date_forecast'])

    x_test.fillna(0, inplace=True)

    label = 'pv_measurement'
    train_data = TabularDataset(x_train)

    tuning_data = TabularDataset(tuning_data)
    thirty_percent_index = int(len(tuning_data) * 0.3)
    tuning_data = tuning_data.iloc[:thirty_percent_index]

    test_data = TabularDataset(x_test)

    predictor = TabularPredictor(label=label,
                                 path="AutoGluonTesting",
                                 eval_metric='mean_absolute_error')

    num_trials = 20  # try at most 5 different hyperparameter configurations for each type of model
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
    preds['date_forecast'] = x_test['date_forecast']
    preds['predicted'] = np.asarray(y_pred)
    preds.to_csv(str(limit) + name + '_' + location + '.csv')
    print('Done with Location: ' + location + "================================================================")


if __name__ == '__main__':
    name = "sec_tuning30_20HPO"
    time_limit = 60 * 60
    do_prediction('A', time_limit, name)
    do_prediction('B', time_limit, name)
    do_prediction('C', time_limit, name)
