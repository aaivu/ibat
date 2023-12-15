import warnings
from typing import Optional

import matplotlib.pyplot as plt
from pandas import DataFrame, to_datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from src.datasets import BUS_654_DWELL_TIMES, BUS_654_RUNNING_TIMES
from src.models.mme4bat import MME4BAT


def run() -> None:
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    rt_df: DataFrame = BUS_654_RUNNING_TIMES.dataframe
    dt_df: DataFrame = BUS_654_DWELL_TIMES.dataframe

    dt_df["date"] = to_datetime(dt_df["date"])
    cutoff_date = to_datetime("2022-09-26")
    dt_df = dt_df.loc[dt_df["date"] < cutoff_date]
    dt_df["week_no"] = dt_df["date"].dt.isocalendar().week
    dt_df["day_of_week"] = dt_df["date"].dt.weekday
    dt_df["time_of_day"] = to_datetime(dt_df["arrival_time"], format="%H:%M:%S").apply(
        lambda x: (x.hour * 60 + x.minute) // 15
    )

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
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
    ]
    new = list(range(1, 37))
    d = dict(zip(old, new))
    dt_df["week_no"] = list(map(lambda x: d[x], dt_df["week_no"]))

    dt_df = dt_df[dt_df["direction"] == 1]
    dt_df = dt_df.drop(dt_df[dt_df["dwell_time_in_seconds"] > 600].index)

    dt_df.sort_values(["trip_id", "bus_stop"], inplace=True)
    dt_df.reset_index(drop=True, inplace=True)
    dt_df.to_csv("del.csv")

    base_model: Optional[MME4BAT] = None
    model: Optional[MME4BAT] = None

    true_prediction = []
    base_model_prediction = []
    model_prediction = []

    dt_df = dt_df[:2000]
    from_idx = 0
    for to_idx in range(1000, len(dt_df), 100):
        dt_chunk: DataFrame = dt_df.iloc[from_idx:to_idx, :].reset_index(drop=True)
        dt_x: DataFrame = dt_chunk[
            ["deviceid", "bus_stop", "day_of_week", "time_of_day"]
        ]
        dt_y: DataFrame = dt_chunk[["dwell_time_in_seconds"]]

        if not model:
            base_model = MME4BAT()
            model = MME4BAT()

            base_model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)
            model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)
        else:
            true_prediction.extend(dt_y["dwell_time_in_seconds"].tolist())
            base_model_prediction.extend(
                base_model.predict(rt_x=None, dt_x=dt_x)["prediction"].tolist()
            )
            model_prediction.extend(
                model.predict(rt_x=None, dt_x=dt_x)["prediction"].tolist()
            )

            model.incremental_fit(
                ni_rt_x=None, ni_rt_y=None, ni_dt_x=dt_x, ni_dt_y=dt_y
            )

        # dt_next_chunk = dt_df.iloc[to_idx: (to_idx + 10), :]
        # dt_x = dt_next_chunk[["deviceid", "bus_stop", "day_of_week", "time_of_day"]]
        # dt_y = dt_next_chunk[["dwell_time_in_seconds"]]

        from_idx = to_idx

    print(
        f"MAE for base model: {mean_absolute_error(true_prediction, base_model_prediction)}"
    )
    print(
        f"MAPE for base model: {mean_absolute_percentage_error(true_prediction, base_model_prediction) * 100}"
    )
    print(
        f"RMSE for base model: {mean_squared_error(true_prediction, base_model_prediction, squared=False)}"
    )
    print("-----------------------------------------------------------------")
    print(
        f"MAE for incremental online learning model: {mean_absolute_error(true_prediction, model_prediction)}"
    )
    print(
        f"""MAPE for incremental online learning model: {
        mean_absolute_percentage_error(true_prediction, model_prediction) * 100
    }"""
    )
    print(
        f"""RMSE for incremental online learning model: {
        mean_squared_error(true_prediction, model_prediction, squared=False)
    }"""
    )

    plt.figure(figsize=(100, 20))

    x = list(range(len(true_prediction)))
    plt.plot(x, true_prediction, label="True dwell time")
    plt.plot(x, base_model_prediction, label="Base model prediction")
    plt.plot(x, model_prediction, label="Prediction with incremental online learning")

    plt.xlabel("X")
    plt.ylabel("Dwell time (s)")
    plt.title("Multiple Lines on a Common Graph")

    plt.legend()
    plt.savefig("dt.png", dpi=150)
