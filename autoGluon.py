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

    predictor.fit(train_data,
                  time_limit=limit,
                  tuning_data=tuning_data, )

    y_pred = predictor.predict(test_data)

    print(y_pred)
    preds = pd.DataFrame()
    preds['date_forecast'] = x_test['date_forecast']
    preds['predicted'] = np.asarray(y_pred)
    preds.to_csv(str(limit) + name + '_' + location + '.csv')
    print('Done with Location: ' + location + "================================================================")


if __name__ == '__main__':
    name = "trying_notVM_"
    time_limit = 5 * 60
    do_prediction('A', time_limit, name)
    do_prediction('B', time_limit, name)
    do_prediction('C', time_limit, name)
