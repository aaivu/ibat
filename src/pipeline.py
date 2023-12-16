import warnings
from typing import Optional, List

import matplotlib.pyplot as plt
from pandas import DataFrame, Timestamp, to_datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from src.datasets import (
    BUS_654_FEATURES_ADDED_DWELL_TIMES,
    BUS_654_FEATURES_ADDED_RUNNING_TIMES,
)
from src.models.use_cases.arrival_time.bus import MME4BAT


def run() -> None:
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    rt_df: DataFrame = BUS_654_FEATURES_ADDED_RUNNING_TIMES.dataframe
    dt_df: DataFrame = BUS_654_FEATURES_ADDED_DWELL_TIMES.dataframe

    numeric_rt_df = rt_df.select_dtypes(include="number")
    numeric_dt_df = dt_df.select_dtypes(include="number")

    base_model: Optional[MME4BAT] = None
    model: Optional[MME4BAT] = None

    true_prediction = []
    base_model_prediction = []
    model_prediction = []

    # dt_df = dt_df[:10000]
    # numeric_dt_df = numeric_dt_df[:10000]

    from_idx = 0
    for to_idx in range(1000, len(numeric_dt_df), 100):
        dt_chunk: DataFrame = numeric_dt_df.iloc[from_idx:to_idx, :].reset_index(
            drop=True
        )
        dt_x: DataFrame = dt_chunk.drop(columns=["dwell_time_in_seconds"])
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

        # dt_next_chunk = numeric_dt_df.iloc[to_idx: (to_idx + 10), :]
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

    # dt_pred_df = dt_df[1000:-100]
    dt_pred_df = dt_df[1000:-83]
    if (
        len(dt_pred_df)
        == len(true_prediction)
        == len(model_prediction)
        == len(base_model_prediction)
    ):
        dt_pred_df["true_prediction"] = true_prediction
        dt_pred_df["model_prediction"] = model_prediction
        dt_pred_df["base_model_prediction"] = base_model_prediction
    else:
        print(
            len(true_prediction),
            len(model_prediction),
            len(base_model_prediction),
            dt_pred_df.shape,
        )
        print("Error: The lengths of the arrays and the DataFrame did not match.")

    # plt.figure(figsize=(100, 20))
    #
    # x = list(range(len(true_prediction)))
    # plt.plot(x, true_prediction, label="True dwell time")
    # plt.plot(x, base_model_prediction, label="Base model prediction")
    # plt.plot(x, model_prediction, label="Prediction with incremental online learning")
    #
    # plt.xlabel("X")
    # plt.ylabel("Dwell time (s)")
    # plt.title("Multiple Lines on a Common Graph")
    #
    # plt.legend()
    # plt.savefig("dt.png", dpi=150)

    time_of_day = 10
    starting_time = Timestamp("10:00:00").time()
    ending_time = Timestamp("10:15:00").time()

    folder_path = "../experiments"

    df = dt_pred_df
    df["date"] = to_datetime(df["date"])
    df["arrival_time"] = to_datetime(df["arrival_time"])
    df["time_of_day"] = df["arrival_time"].dt.hour

    directions = [1, 2]
    bus_stops = df["bus_stop"].unique()

    for direction in directions:
        for bus_stop in bus_stops:
            filtered_df = df[
                (df["direction"] == direction)
                & (df["bus_stop"] == bus_stop)
                & (df["time_of_day"] == time_of_day)
                & (df["arrival_time"].dt.time >= starting_time)
                & (df["arrival_time"].dt.time < ending_time)
            ]

            print(bus_stop, filtered_df.shape)

            if len(filtered_df) > 0:
                export_mean_dt_plot_as_image(
                    df=filtered_df,
                    bus_stop=bus_stop,
                    starting_time=starting_time,
                    ending_time=ending_time,
                    export_at=folder_path,
                )


def export_mean_dt_plot_as_image(
    df: DataFrame,
    bus_stop: int,
    starting_time,
    ending_time,
    export_at: str,
) -> None:
    dt_in_seconds_df = (
        df.groupby("date")["dwell_time_in_seconds"]
        .mean()
        .reset_index()
        .sort_values(by="date")
    )

    base_model_prediction_df = (
        df.groupby("date")["base_model_prediction"]
        .mean()
        .reset_index()
        .sort_values(by="date")
    )

    model_prediction_df = (
        df.groupby("date")["model_prediction"]
        .mean()
        .reset_index()
        .sort_values(by="date")
    )

    x = model_prediction_df["date"]

    plt.figure(figsize=(13, 5))
    plt.plot(
        x,
        dt_in_seconds_df["dwell_time_in_seconds"],
        label="True dwell time",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        x,
        base_model_prediction_df["base_model_prediction"],
        label="Base model prediction",
        marker="o",
        linestyle="--",
        color="green",
    )
    plt.plot(
        x,
        model_prediction_df["model_prediction"],
        label="Prediction with incremental online learning",
        marker="o",
        linestyle="--",
        color="orange",
    )
    plt.xlabel("Date")
    plt.ylabel("Mean dwell time")
    plt.title(
        f"The mean dwell time at bus stop {bus_stop} for each day between {starting_time} and {ending_time}."
    )
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_at}/dt/{bus_stop}.png", dpi=200)
