from build_pandas import build_pandas
#import xgboost as xgb
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import utils
import numpy as np


def dothing(location):
    x_train, tuning, x_test = utils.preprocess_category_estimated_observed(location)
    x_train.drop(["time", "snow_density:kgm3"], axis=1, inplace=True)
    x_test.drop(["snow_density:kgm3"], axis=1, inplace=True)
    x_train['date_forecast'] = (pd.to_datetime(x_train['date_forecast'], format='%Y') - pd.to_datetime('2000', format='%Y')).dt.total_seconds()
    x_test['date_forecast_dt'] = x_test['date_forecast']
    x_test['date_forecast'] = (pd.to_datetime(x_test['date_forecast'], format='%Y') - pd.to_datetime('2000', format='%Y')).dt.total_seconds()

    thirty_percent_index = int(len(tuning) * 0.4)
    tuning_data = tuning.iloc[:thirty_percent_index]

    h2o.init()

    aml = H2OAutoML(max_models=20, seed=1, max_runtime_secs=1200)
    aml.train(x=list(x_train.drop('pv_measurement', axis=1).columns), y='pv_measurement',
              training_frame=h2o.H2OFrame(x_train))

    lb = aml.leaderboard
    lb.head(rows=lb.nrows)

    print('Leader:-----------------------------------------------------------   ')
    print(aml.leader)

    preds = aml.predict(h2o.H2OFrame(x_test))
    print(preds)
    y_pred = h2o.as_list(preds)

    preds = pd.DataFrame()
    preds['date_forecast'] = x_test['date_forecast_dt']
    preds['predicted'] = np.asarray(y_pred)
    preds.to_csv('h2o_no_estimated_' + location + '.csv', index=False)


dothing('A')
dothing('B')
dothing('C')
