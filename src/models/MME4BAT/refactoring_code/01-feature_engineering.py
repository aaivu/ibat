import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, date
import glob
import os


path = "./cleaned/data/bus_dwell_times_654.csv"
stop_times = pd.read_csv(path)

stop_times["date"] = pd.to_datetime(stop_times["date"])
cutoff_date = pd.to_datetime("2022-03-01")
stop_times = stop_times.loc[stop_times["date"] < cutoff_date]
stop_times["week_no"] = stop_times["date"].dt.isocalendar().week

stop_times["day_of_week"] = stop_times["date"].dt.weekday
stop_times["time_of_day"] = list(
    map(lambda x: x.hour, pd.to_datetime(stop_times["arrival_time"]))
)

df = stop_times
old = [
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
]
new = list(range(1, 24))
d = dict(zip(old, new))
df["week_no"] = list(map(lambda x: d[x], df["week_no"]))


df = df[df["direction"] == 1]

long_stops = [101, 105, 109, 113]
dfl = df.loc[df["bus_stop"].isin(long_stops)]

short_stops = [102, 106, 107, 108, 110, 111, 112, 114]
dfs = df.loc[df["bus_stop"].isin(short_stops)]

mean = np.mean(dfs["dwell_time_in_seconds"], axis=0)
sd = np.std(dfs["dwell_time_in_seconds"], axis=0)

# print(mean)

# print(sd)

df = df.drop(df[df["dwell_time_in_seconds"] > 600].index)

# def fill_nan_median(ts, medians):
#   ix = pd.to_datetime(ts.index)
#   ts.index = pd.to_datetime(ix.dayofweek * 24 * 60 * 60 + x.hour * 60 * 60 + x.minute * 60, unit = 's')
#   ts = ts.fillna(medians)
#   ts.index = ix
#   return ts

dft = df.groupby("week_no")

groupings = list(dft.groups.keys())

df.reset_index(drop=True, inplace=True)

for i in range(3, len(groupings)):
    curr = dft.get_group(groupings[i])
    prev1 = dft.get_group(groupings[i - 1])
    prev2 = dft.get_group(groupings[i - 2])
    prev3 = dft.get_group(groupings[i - 3])

    # curr['dt(t-1)']= prev['day_of_week']
    for index, row in curr.iterrows():
        day = row["day_of_week"]
        time = row["time_of_day"]
        stop = row["bus_stop"]
        agg1 = prev1.loc[
            (prev1["day_of_week"] == day)
            & (prev1["time_of_day"] == time)
            & (prev1["bus_stop"] == stop)
        ]
        agg2 = prev2.loc[
            (prev2["day_of_week"] == day)
            & (prev2["time_of_day"] == time)
            & (prev2["bus_stop"] == stop)
        ]
        agg3 = prev3.loc[
            (prev3["day_of_week"] == day)
            & (prev3["time_of_day"] == time)
            & (prev3["bus_stop"] == stop)
        ]

        df.at[index, "dt(w-1)"] = round(agg1["dwell_time_in_seconds"].mean(), 1)
        df.at[index, "dt(w-2)"] = round(agg2["dwell_time_in_seconds"].mean(), 1)
        df.at[index, "dt(w-3)"] = round(agg3["dwell_time_in_seconds"].mean(), 1)

df["dt(w-1)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)
df["dt(w-2)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)
df["dt(w-3)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)

for name, group in df.groupby("date"):
    for index, row in group.iterrows():
        time = row["time_of_day"]
        stop = row["bus_stop"]

        df.at[index, "dt(t-1)"] = round(
            group["dwell_time_in_seconds"][
                (group["time_of_day"] == (time - 1)) & (group["bus_stop"] == stop)
            ].mean(),
            1,
        )
        df.at[index, "dt(t-2)"] = round(
            group["dwell_time_in_seconds"][
                (group["time_of_day"] == (time - 2)) & (group["bus_stop"] == stop)
            ].mean(),
            1,
        )

df["dt(t-1)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)
df["dt(t-2)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)

for name, group in df.groupby("trip_id"):
    for index, row in group.iterrows():
        stop = row["bus_stop"]
        trip = row["trip_id"]
        df.at[index, "dt(n-1)"] = round(
            group["dwell_time_in_seconds"][(group["bus_stop"] == (stop - 1))].mean(), 1
        )
        df.at[index, "dt(n-2)"] = round(
            group["dwell_time_in_seconds"][(group["bus_stop"] == (stop - 2))].mean(), 1
        )
        df.at[index, "dt(n-3)"] = round(
            group["dwell_time_in_seconds"][(group["bus_stop"] == (stop - 3))].mean(), 1
        )

df["dt(n-1)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)
df["dt(n-2)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)
df["dt(n-3)"].fillna(
    df.groupby(["bus_stop", "time_of_day"])["dwell_time_in_seconds"].transform("mean"),
    inplace=True,
)

df[
    [
        "dt(w-1)",
        "dt(w-2)",
        "dt(w-3)",
        "dt(t-1)",
        "dt(t-2)",
        "dt(n-1)",
        "dt(n-2)",
        "dt(n-3)",
    ]
] = df[
    [
        "dt(w-1)",
        "dt(w-2)",
        "dt(w-3)",
        "dt(t-1)",
        "dt(t-2)",
        "dt(n-1)",
        "dt(n-2)",
        "dt(n-3)",
    ]
].apply(
    pd.Series.round
)

filename = "bus_stop_times_feature_added_new.csv"
data.to_csv(filename, encoding="utf-8-sig", index=False)

# train = df[df['week_no']<20]
# test = df[df['week_no']>19]

# np.corrcoef(df['dwell_time_in_seconds'],df['dt(w-1)'])

# np.corrcoef(df['dwell_time_in_seconds'],df['dt(t-1)'])

# np.corrcoef(df['dwell_time_in_seconds'],df['dt(w-2)'])

# X = dfs[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']]

# X.reset_index(drop = True, inplace = True)

# X

# ohd=pd.get_dummies(X,columns=['bus_stop','deviceid'])

# X = ohd

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# #scaled = scaler.fit_transform(X[['dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']])

# X= pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=X[['dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']])

# #scaled = pd.DataFrame(scaled)

# #scaled

# X

# #X = pd.concat([X, scaled], axis=1)

# X.columns

# X = X.drop(columns = ['dt(t-1)',
#              'dt(t-2)',       'dt(w-1)',       'dt(w-2)',       'dt(w-3)',
#              'dt(n-1)',       'dt(n-2)',       'dt(n-3)'])

# X

# X = X.drop(columns=['day_of_week'])

# X.columns = X.columns.astype(str)

# X1 = df[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day']]

# X1

# X.columns

# X_train= train[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']]
# X_test = test[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']]

# X1_train = train[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day']]
# X1_test = test[['deviceid','bus_stop','day_of_week','weekday/end','time_of_day']]

# feature_names =  ['deviceid','bus_stop','day_of_week','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']

# """for dwelltime at one bus stop"""

# X_train= train[['deviceid','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']]
# X_test = test[['deviceid','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']]

# X1_train = train[['deviceid','day_of_week','weekday/end','time_of_day']]
# X1_test = test[['deviceid','day_of_week','weekday/end','time_of_day']]

# feature_names =  ['deviceid','weekday/end','time_of_day','dt(t-1)','dt(t-2)','dt(w-1)','dt(w-2)','dt(w-3)','dt(n-1)','dt(n-2)','dt(n-3)']

# X1_test

# y = dfs[['dwell_time_in_seconds']]

# y_train= train[['dwell_time_in_seconds']]
# y_test= test[['dwell_time_in_seconds']]

# y

# y.reset_index(drop = True, inplace = True)

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold

# from sklearn import linear_model
# lr = linear_model.LinearRegression()
# scores = cross_val_score(lr, X, y, scoring='r2', cv=KFold(n_splits=10,shuffle=False )) # shuffle=False
# rmse = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=10,shuffle=False)) #
# rmse = np.sqrt(list(-rmse))
# lr_r2 = scores.mean()
# lr_rmse = rmse.mean()

# print('lr_r2 =' + str(lr_r2))
# print('lr_rmse =' + str(lr_rmse))

# from sklearn.svm import SVR
# svr = SVR(kernel = 'rbf')
# scores = cross_val_score(svr, X, y, scoring='r2', cv=KFold(n_splits=8, shuffle=False))
# rmse = cross_val_score(svr, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=8, shuffle=False))
# rmse = np.sqrt(list(-rmse))
# svr_r2 = scores.mean()
# svr_rmse = rmse.mean()

# print('svr_r2 =' + str(svr_r2))
# print('svr_rmse =' + str(svr_rmse))

# from sklearn.tree import DecisionTreeRegressor
# dt = DecisionTreeRegressor(random_state=0)
# scores = cross_val_score(dt, X, y, scoring='r2', cv=KFold(n_splits=5,shuffle=False))
# rmse = cross_val_score(dt, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=5,shuffle=False))
# rmse = np.sqrt(list(-rmse))
# dt_r2 = scores.mean()
# dt_rmse = rmse.mean()

# print('dt_r2 =' + str(dt_r2))
# print('dt_rmse =' + str(dt_rmse))

# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor(n_estimators = 100,max_depth = 5, random_state = 42)
# scores = cross_val_score(rfr, X, y, scoring='r2', cv=KFold(n_splits=5,shuffle=False)) #,
# rmse = cross_val_score(rfr, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=5,shuffle=False)) #,
# rmse = np.sqrt(list(-rmse))
# rfr_r2 = scores.mean()
# rfr_rmse = rmse.mean()

# print('rfr_r2 =' + str(rfr_r2))
# print('rfr_rmse =' + str(rfr_rmse))

# import xgboost as xg
# xgb= xg.XGBRegressor(n_estimators = 100)
# scores = cross_val_score(xgb, X, y, scoring='r2', cv=KFold(n_splits=5,shuffle=False))
# rmse = cross_val_score(xgb, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=5,shuffle=False))
# rmse = np.sqrt(list(-rmse))
# xgb_r2 = scores.mean()
# xgb_rmse = rmse.mean()

# print('xgb_r2 =' + str(xgb_r2))
# print('xgb_rmse =' + str(xgb_rmse))

# param={'learning_rate':[0.01,0.1,1].
#        'max_depth':[]}

# rs =

# xgb.fit(X_train,y_train)

# xgb.feature_importances_

# plt.barh(feature_names, xgb.feature_importances_)

# !pip install shap
# import shap
# explainer = shap.TreeExplainer(xgb)
# shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test, plot_type="bar")

# from sklearn.inspection import permutation_importance

# perm_importance = permutation_importance(xgb, X_test, y_test)

# plt.barh(feature_names, perm_importance.importances_mean)
# plt.xlabel("Permutation Importance")

# def correlation_heatmap(train):
#     correlations = train.corr()

#     fig, ax = plt.subplots(figsize=(10,10))
#     sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
#                 square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
#                 )
#     plt.show();

# correlation_heatmap(X[feature_names])

# !pip install boruta

# from boruta import BorutaPy

# feature_selector = BorutaPy(
#     verbose=2,
#     estimator=rfr,
#     n_estimators='auto',
#     max_iter=10,
#     random_state=42,
# )

# feature_selector.fit(np.array(X), np.array(y))

# print("\n------Support and Ranking for each feature------")
# for i in range(len(feature_selector.support_)):
#     if feature_selector.support_[i]:
#         print("Passes the test: ", X.columns[i],
#               " - Ranking: ", feature_selector.ranking_[i])
#     else:
#         print("Doesn't pass the test: ",
#               X.columns[i], " - Ranking: ", feature_selector.ranking_[i])

# X_filtered=feature_selector.transform(np.array (X))

# scores = cross_val_score(rfr, X_filtered, y, scoring='r2', cv=KFold(n_splits=4, shuffle=False))
# rmse = cross_val_score(rfr, X_filtered, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=4, shuffle=False))
# rmse = np.sqrt(list(-rmse))
# rfr_r2 = scores.mean()
# rfr_rmse = rmse.mean()

# print('rfr_r2 =' + str(rfr_r2))
# print('rfr_rmse =' + str(rfr_rmse))

# pip install catboost

# from catboost import Pool, CatBoostRegressor

# cbr = CatBoostRegressor(iterations=2,
#                           depth=2,
#                           learning_rate=1,
#                           loss_function='RMSE')

# scores = cross_val_score(cbr, X, y, scoring='r2', cv=KFold(n_splits=4, shuffle=False))
# rmse = cross_val_score(cbr, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=4, shuffle=False))
# rmse = np.sqrt(list(-rmse))
# cbr_r2 = scores.mean()
# cbr_rmse = rmse.mean()

# print('cbr_r2 =' + str(cbr_r2))
# print('cbr_rmse =' + str(cbr_rmse))

# !pip install lightgbm

# import lightgbm as lgb
# hyper_params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': ['l1','l2'],
#     'learning_rate': 0.005,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 10,
#     'verbose': 0,
#     "max_depth": 8,
#     "num_leaves": 128,
#     "max_bin": 512,
#     "num_iterations": 100000
# }

# lgb=lgb.LGBMRegressor(**hyper_params)
# scores = cross_val_score(lgb, X, y, scoring='r2', cv=KFold(n_splits=4, shuffle=False))
# rmse = cross_val_score(lgb, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=4, shuffle=False))
# rmse = np.sqrt(list(-rmse))
# lgb_r2 = scores.mean()
# lgb_rmse = rmse.mean()

# print('lgb_r2 =' + str(lgb_r2))
# print('lgb_rmse =' + str(lgb_rmse))


# df = stop_times.append([trip_ends])


# df['direction'] = df.groupby('trip_id')['direction'].ffill().bfill()

# df
