import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from ibat.concept_drift_detector import CDD
from ibat.concept_drift_detector.strategies import DDM


def split_train_test(split_date, df):
    before_missing_dates = df[df["date"] < split_date]

    after_missing_dates = df[df["date"] >= split_date]

    before_missing_dates.reset_index(drop=True, inplace=True)

    after_missing_dates.reset_index(drop=True, inplace=True)

    return before_missing_dates, after_missing_dates


sorted_mean_arrival_time = pd.read_csv("../../asserts/datasets/input.csv")

train, test = split_train_test("2022-03-01", sorted_mean_arrival_time)

X_train = train.drop(
    columns=[
        "arrival_time_in_seconds",
        "date",
        "Unnamed: 0",
    ]
)

X_test = test.drop(
    columns=[
        "arrival_time_in_seconds",
        "date",
        "Unnamed: 0",
    ]
)

y_train = train[["arrival_time_in_seconds"]]

y_test = test[["arrival_time_in_seconds"]]

pipeline = Pipeline([("model", XGBRegressor(objective="reg:squarederror"))])

pipeline.fit(X=X_train, y=y_train)

strategy = DDM(
    warning_level=0.5,
    drift_level=1,
    min_num_instances=5,
)

cdd = CDD(strategy=strategy)

is_detected = cdd.is_concept_drift_detected(pipeline, X_test, y_test)

print(is_detected)
