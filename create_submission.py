import pandas as pd

file_name = 'raw_predictions/ABC_'

pred_a = pd.read_csv(file_name + 'A.csv')
pred_a['date'] = pd.to_datetime(pred_a['date_forecast'])
pred_b = pd.read_csv(file_name + 'B.csv')
pred_b['date'] = pd.to_datetime(pred_b['date_forecast'])
pred_c = pd.read_csv(file_name + 'C.csv')
pred_c['date'] = pd.to_datetime(pred_c['date_forecast'])

test = pd.read_csv('data/test.csv')
test['time'] = pd.to_datetime(test['time'])

submission = pd.DataFrame(columns=['prediction'])

for val in pred_a['date']:
    if test['time'].eq(val).any():
        row = pred_a.loc[pred_a['date'] == val]
        submission = submission._append({'prediction': float(row['predicted'])}, ignore_index=True)

for val in pred_b['date']:
    if test['time'].eq(val).any():
        row = pred_b.loc[pred_a['date'] == val]
        submission = submission._append({'prediction': float(row['predicted'])}, ignore_index=True)

for val in pred_c['date']:
    if test['time'].eq(val).any():
        row = pred_c.loc[pred_a['date'] == val]
        submission = submission._append({'prediction': float(row['predicted'])}, ignore_index=True)

submission['prediction'] = submission['prediction'].where(submission['prediction'] >= 0, 0)
submission.index.name = "id"

submission.to_csv('submission_' + 'attempt_2.csv')
