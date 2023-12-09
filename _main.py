import pandas as pd
import xgboost as xg
from sklearn.pipeline import Pipeline
# from src.concept_drift_detector import CDD
from src.concept_drift_detector.concept_drift_detector import  CDD
from src.concept_drift_detector.strategies.page_hinkley import PageHinkley
from src.concept_drift_detector.strategies.adwin import Adwin
import numpy as np

def split_train_test(split_date ,df):
    before_missing_dates = df[df['date'] < split_date]
    after_missing_dates = df[df['date'] >= split_date]
    before_missing_dates.reset_index(drop=True, inplace=True)
    after_missing_dates.reset_index(drop=True, inplace=True)
    return before_missing_dates, after_missing_dates

sorted_mean_arrival_time = pd.read_csv('src/concept_drift_detector/input.csv')
# train, test = split_train_test('2022-03-01', sorted_mean_arrival_time)

# X_train = train.drop(columns=['arrival_time_in_seconds', 'date'])
# X_test = test.drop(columns=['arrival_time_in_seconds', 'date'])
# y_train = train['arrival_time_in_seconds']
# y_test = test['arrival_time_in_seconds']

# pipeline = Pipeline(  [(   "model", xg.XGBRegressor(objective ='reg:linear')   )]   )
# pipeline.fit(X=X_train, y=y_train)

stream = np.array(list(sorted_mean_arrival_time['arrival_time_in_seconds']))

strategy = Adwin(delta=0.2)
# strategy = PageHinkley(threshold = 6500)
cdd = CDD(strategy)
# is_detected = cdd.is_concept_drift_detected(pipeline, X_test, y_test)
is_detected = cdd.is_concept_drift_detected(None, stream, None)
print(is_detected)