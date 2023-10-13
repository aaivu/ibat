import pandas as pd
import xgboost as xg
from concept_drift_detector import CDD 
from sklearn.pipeline import Pipeline


def split_train_test(split_date ,df):
    before_missing_dates = df[df['date'] < split_date]
    after_missing_dates = df[df['date'] >= split_date]

    before_missing_dates.reset_index(drop=True, inplace=True)
    after_missing_dates.reset_index(drop=True, inplace=True)
    
    return before_missing_dates, after_missing_dates



sorted_mean_arrival_time = pd.read_csv('src\concept_drift_detector\input.csv')
train,test = split_train_test('2022-03-01' ,sorted_mean_arrival_time)

X_train = train.drop(columns=['arrival_time_in_seconds', 'date'])
X_test = test.drop(columns=['arrival_time_in_seconds', 'date'])
y_train = train['arrival_time_in_seconds']
y_test = test['arrival_time_in_seconds']

pipeline = Pipeline(  [(   "model", xg.XGBRegressor(objective ='reg:linear')   )]   )
pipeline.fit(X=X_train, y=y_train)


cd = CDD(pipeline, X_test,y_test)

is_detect = cd.is_concept_drift(pipeline, X_test,y_test)

print(is_detect)
