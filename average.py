import pandas as pd

name1 = 'submissions/Average_of_two.csv'
name2 = 'submission_3600testing_small_train_tuning_attempt_2.csv'
name3 = ''
name4 = ''

df1 = pd.read_csv(name1)
df2 = pd.read_csv(name2)
df3 = pd.read_csv('submissions/averaged_predictions.csv')
df4 = pd.read_csv('submissions/averaged_predictions.csv')

average_predicted = (df1['prediction'] + df2['prediction']) / 2
average_predicted.index.name = "id"
average_predicted.to_csv(name1.split('.')[0] + '_and_' + name2)
