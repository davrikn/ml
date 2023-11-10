import pandas as pd

name1 = 'submission_1_tuning_40HPO__30_attempt_2.csv'
name2 = 'submission_3600sec_40tuning_noHPO_attempt.csv'
name3 = 'submission_0_tuning_noHPO_10_attempt_2.csv'
name4 = 'submission_3600testing_small_train_tuning_attempt_2.csv'

df1 = pd.read_csv(name1)
df2 = pd.read_csv(name2)
df3 = pd.read_csv(name3)
df4 = pd.read_csv(name4)

average_predicted = (df1['prediction'] + df2['prediction'] + df3['prediction'] + df4['prediction']) / 4
average_predicted.index.name = "id"
average_predicted.to_csv(name1.split('.')[0] + '_and_' + name2.split('.')[0]+ '_and_' + name3.split('.')[0]+ '_and_' + name4)
