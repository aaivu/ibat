import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, date

import xgboost as xgb, DMatrix
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV


path = "./cleaned/data/bus_stop_times_feature_added_all.csv"
df = pd.read_csv(path)


def condition(x):
    if x == 0:
        return 0
    else:
        return 1


df["dwell/pass"] = df["dwell_time_in_seconds"].apply(condition)

test = df[df["week_no"] >= 16]
train = df[df["week_no"] < 16]

test.reset_index(drop=True, inplace=True)

xgb = xgb.XGBClassifier()

# Xtrain = train[['deviceid','bus_stop','day_of_week', 'Sunday/holiday', 'saturday','time_of_day','dt(w-1)', 'dt(w-2)', 'dt(w-3)', 'dt(t-1)','dt(t-2)', 'dt(n-1)', 'dt(n-2)', 'dt(n-3)','temp', 'precip','rt(n-1)']]
# ytrain = train[['dwell/pass']]

# Xtest = test[['deviceid','bus_stop','day_of_week', 'Sunday/holiday', 'saturday','time_of_day','dt(w-1)', 'dt(w-2)', 'dt(w-3)', 'dt(t-1)','dt(t-2)', 'dt(n-1)', 'dt(n-2)', 'dt(n-3)','temp', 'precip','rt(n-1)']]
# ytest = test[['dwell/pass']]

Xtrain = train[
    [
        "deviceid",
        "bus_stop",
        "day_of_week",
        "time_of_day",
        "dt(w-1)",
        "dt(w-2)",
        "dt(w-3)",
        "dt(t-1)",
        "dt(t-2)",
        "dt(n-1)",
        "dt(n-2)",
    ]
]
ytrain = train[["dwell/pass"]]

Xtest = test[
    [
        "deviceid",
        "bus_stop",
        "day_of_week",
        "time_of_day",
        "dt(w-1)",
        "dt(w-2)",
        "dt(w-3)",
        "dt(t-1)",
        "dt(t-2)",
        "dt(n-1)",
        "dt(n-2)",
    ]
]
ytest = test[["dwell/pass"]]

xgb.fit(Xtrain, ytrain)
pred_xg = xgb.predict(Xtest)

print(f"accuracy_score: {accuracy_score(ytest, pred_xg)}")
print(f"f1_score: {f1_score(ytest, pred_xg)}")

print(confusion_matrix(ytest, pred_xg))

# pred_xgt = xgb.predict(Xtrain)

# accuracy_score(ytrain, pred_xgt)

# pred_xg = pd.Series(pred_xg, name='XGBoost_class')
# pred =test.merge(pred_xg,left_index=True, right_index=True)
# pred

# pred['XGBoost_class'].value_counts()

# train = train[train['dwell/pass']!=0]
# test_r = pred[pred['XGBoost_class']!=0]

# test_r

# import xgboost as xgb
# xgb=  xgb.XGBRegressor(colsample_bytree = 0.7, learning_rate = 0.1,max_depth = 6, alpha = 10, n_estimators = 1000)

# Xtrain = train[['deviceid','bus_stop','day_of_week', 'Sunday/holiday', 'saturday','time_of_day','dt(w-1)', 'dt(w-2)', 'dt(w-3)', 'dt(t-1)','dt(t-2)', 'dt(n-1)', 'dt(n-2)', 'dt(n-3)','temp', 'precip','rt(n-1)']]
# ytrain = train[['dwell_time_in_seconds']]

# Xtest = test_r[['deviceid','bus_stop','day_of_week', 'Sunday/holiday', 'saturday','time_of_day','dt(w-1)', 'dt(w-2)', 'dt(w-3)', 'dt(t-1)','dt(t-2)', 'dt(n-1)', 'dt(n-2)', 'dt(n-3)','temp', 'precip','rt(n-1)']]
# ytest = test_r[['dwell_time_in_seconds']]

# xgb.fit(Xtrain,ytrain)
# pred_xg_r = xgb.predict(Xtest)
# rmse = np.sqrt(mean_squared_error(ytest, pred_xg_r))

# print("RMSE (1): %f" % (rmse))

# mape = mean_absolute_percentage_error(ytest, pred_xg_r)
# print("MAPE (1): %f" % (mape))

# mae = mean_absolute_error(ytest, pred_xg_r)
# print("MAE (1): %f" % (mae))


# r2 = r2_score(ytest, pred_xg_r)
# print("r2 (1): %f" % (r2))

# test_r.reset_index(drop = True, inplace = True)

# pred_xg_r = pd.Series(pred_xg_r, name='XGBoost_reg')
# pred_r =test_r.merge(pred_xg_r,left_index=True, right_index=True)
# pred_r

# pred_r.drop(['XGBoost_class'], axis=1,inplace=True)

# pred_r

# pred_r.rename(columns = {'XGBoost_reg':'XGBoost_class'}, inplace = True)

# pred_r

# pred_c = pred[pred['XGBoost_class']==0]

# pred_dwell = pd.concat([pred_c, pred_r])

# pred_dwell = pred_dwell.sort_values(['trip_id', 'bus_stop'])

# pred_dwell.reset_index(drop = True, inplace = True)

# pred_dwell

# rmse = np.sqrt(mean_squared_error(pred_dwell['dwell_time_in_seconds'], pred_dwell['XGBoost_class']))

# print("RMSE (1): %f" % (rmse))

# mape = mean_absolute_percentage_error(pred_dwell['dwell_time_in_seconds'], pred_dwell['XGBoost_class'])
# print("MAPE (1): %f" % (mape))

# mae = mean_absolute_error(pred_dwell['dwell_time_in_seconds'], pred_dwell['XGBoost_class'])
# print("MAE (1): %f" % (mae))


# r2 = r2_score(pred_dwell['dwell_time_in_seconds'], pred_dwell['XGBoost_class'])
# print("r2 (1): %f" % (r2))

# (pred_dwell['XGBoost_class'] < 0).sum()

# pred = pred_dwell

# pred

# pred['DateTime'] = pd.to_datetime(pred['date'] + ' ' + pred['arrival_time'])
# ref_freq = '15min'
# ix = pd.DatetimeIndex(pd.to_datetime(pred['DateTime'])).floor(ref_freq)
# pred["DateTimeRef"] = ix

# path= '/content/drive/Shareddrives/MSc - Shiveswarran/Predicted values/predicted_dwell_times/predicted_dwell_times.csv'

# pred_dwell = pd.read_csv(path)

# pred = pred[pred['DateTimeRef'].isin(pred_dwell['DateTimeRef'].tolist())]

# pred = pred.sort_values(['trip_id', 'bus_stop'])

# pred.reset_index(drop = True, inplace = True)

# pred_dwell = pred_dwell.sort_values(['trip_id', 'bus_stop'])

# pred_dwell.reset_index(drop = True, inplace = True)

# pred_dwell =pred_dwell.merge(pred['XGBoost_class'],left_index=True, right_index=True)

# def download_csv(data,filename):
#   filename= filename + '.csv'
#   data.to_csv(filename, encoding = 'utf-8-sig',index= False)
#   files.download(filename)

# download_csv(pred_dwell,'predicted_dwell_times')

# ytest

# pred_xg

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier()

# knn.fit(Xtrain,ytrain)
# pred_knn = knn.predict(Xtest)

# accuracy_score(ytest, pred_knn)

# confusion_matrix(ytest, pred_knn)

# import xgboost as xgb
# xgb=  xgb.XGBRegressor()
# param_grid = {
#     "max_depth": [3, 5, 7],
#     "learning_rate": [0.1, 0.01, 0.05],
#     #"gamma": [0, 0.25, 1],
#     #"reg_lambda": [0, 0.2, 1],
#     #"subsample": [0.8,1],
#     #"colsample_bytree": [0.5,1],
# }

# grid = GridSearchCV(xgb.XGBRFRegressor(), param_grid, refit = True, verbose = 3, n_jobs=-1)

# xgb.fit(Xtrain,ytrain)
# pred_xg = xgb.predict(Xtest)
