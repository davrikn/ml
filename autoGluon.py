from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import ut
import numpy as np
import random
import string


def do_prediction(location, time_limit):
    x_train, x_test = ut.preprocess_category(location)
    x_train.drop(["time"], axis=1, inplace=True)

    x_train['date_forecast'] = pd.to_datetime(x_train['date_forecast'])

    x_test.fillna(0, inplace=True)

    label = 'pv_measurement'
    train_data = TabularDataset(x_train)

    test_data = TabularDataset(x_test)

    predictor = TabularPredictor(label=label,
                                 path="AutoGluonTesting",
                                 eval_metric='mean_absolute_error')

    predictor.fit(train_data,
                  time_limit=time_limit,
                  presets=['high_quality'])

    y_pred = predictor.predict(test_data)

    print(y_pred)
    preds = pd.DataFrame()
    preds['date_forecast'] = x_test['date_forecast']
    preds['predicted'] = np.asarray(y_pred)
    random_string = ''.join(random.choice(string.ascii_letters) for _ in range(4))
    preds.to_csv(location + str(time_limit) + random_string + '.csv')


time_limit = 30 * 60
do_prediction('A', time_limit)
do_prediction('B', time_limit)
do_prediction('C', time_limit)
