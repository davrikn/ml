import pandas as pd

df1 = pd.read_csv('submissions/averaged_predictions.csv')
df2 = pd.read_csv('submissions/average_prediction.csv')
df3 = pd.read_csv('submissions/averaged_predictions.csv')
df4 = pd.read_csv('submissions/averaged_predictions.csv')

average_predicted = (df1['prediction'] + df2['prediction']) / 2
average_predicted.to_csv('Average_of_two.csv')
