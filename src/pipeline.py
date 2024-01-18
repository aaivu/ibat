import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
from pandas import concat, DataFrame, to_datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from src.concept_drift_detector.strategies import IStrategy
from src.datasets import (
    BUS_654_FEATURES_ADDED_DWELL_TIMES,
    BUS_654_FEATURES_ADDED_RUNNING_TIMES,
)
from src.models.use_cases.arrival_time.bus import MME4BAT


def run_exp(
    hist_start: datetime,
    hist_end: datetime,
    stream_start: datetime,
    stream_end: datetime,
    interval_min: float,
    active_strategy: Optional[bool] = False,
    cdd_strategy: Optional[IStrategy] = None,
    output_parent_dir: Optional[str] = "./",
    label: Optional[str] = "",
) -> None:
    """
    Run the experiment.

    Args:
        hist_start: Start timestamp (inclusive) for historical data.
        hist_end: End timestamp (exclusive) for historical data.
        stream_start: Start timestamp (inclusive) for streaming data.
        stream_end: End timestamp (exclusive) for streaming data.
        interval_min: Time interval (in minutes) for data processing.
        active_strategy: Flag indicating whether to use active strategy (default is False).
        cdd_strategy: Strategy to be used for detecting concept drift. Required if active_strategy is True.
        output_parent_dir: Parent directory's path to save experiment results.
        label: Experiment label (default is an empty string).

    Returns:
        None
    """
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    if active_strategy and cdd_strategy is None:
        raise ValueError("cdd_strategy must be provided when active_strategy is True.")

    rt_df: DataFrame = BUS_654_FEATURES_ADDED_RUNNING_TIMES.dataframe
    dt_df: DataFrame = BUS_654_FEATURES_ADDED_DWELL_TIMES.dataframe

    rt_df["start_datetime"] = to_datetime(rt_df["date"] + " " + rt_df["start_time"])
    dt_df["arrival_datetime"] = to_datetime(dt_df["date"] + " " + dt_df["arrival_time"])

    base_model: Optional[MME4BAT] = None
    model: Optional[MME4BAT] = None

    result_rt_df = DataFrame(
        columns=rt_df.columns.tolist()
                + ["true_prediction", "base_model_prediction", "model_prediction"]
    )

    result_dt_df = DataFrame(
        columns=dt_df.columns.tolist()
                + ["true_prediction", "base_model_prediction", "model_prediction"]
    )
    rt_true_predictions = []
    rt_base_model_predictions = []
    rt_model_predictions = []
    dt_true_predictions = []
    dt_base_model_predictions = []
    dt_model_predictions = []

    rt_x_buffer: Optional[DataFrame] = None
    rt_y_buffer: Optional[DataFrame] = None
    dt_x_buffer: Optional[DataFrame] = None
    dt_y_buffer: Optional[DataFrame] = None

    from_date_time = hist_start
    to_date_time = hist_end

    while from_date_time < stream_end:
        print(
            f"DATA STREAM: [{from_date_time.strftime('%Y-%m-%d %H:%M:%S')} - {to_date_time.strftime('%Y-%m-%d %H:%M:%S')})",
            end="",
            flush=True,
        )

        rt_chunk: DataFrame = rt_df.loc[
                              (from_date_time <= rt_df["start_datetime"])
                              & (rt_df["start_datetime"] < to_date_time),
                              :,
                              ].reset_index(drop=True)

        dt_chunk: DataFrame = dt_df.loc[
                              (from_date_time <= dt_df["arrival_datetime"])
                              & (dt_df["arrival_datetime"] < to_date_time),
                              :,
                              ].reset_index(drop=True)

        from_date_time = stream_start if from_date_time == hist_start else to_date_time
        to_date_time = from_date_time + timedelta(minutes=interval_min)

        if len(rt_chunk) == 0:
            print(" | NO INSTANCES FOR RUNNING TIME")
            continue

        if len(dt_chunk) == 0:
            print(" | NO INSTANCES FOR DWELL TIME")
            continue

        numeric_rt_chunk = rt_chunk.select_dtypes(include="number")
        numeric_dt_chunk = dt_chunk.select_dtypes(include="number")

        rt_x: DataFrame = numeric_rt_chunk.drop(columns=["run_time_in_seconds"])
        rt_y: DataFrame = numeric_rt_chunk[["run_time_in_seconds"]]
        dt_x: DataFrame = numeric_dt_chunk.drop(columns=["dwell_time_in_seconds"])
        dt_y: DataFrame = numeric_dt_chunk[["dwell_time_in_seconds"]]

        if not model:
            base_model = MME4BAT()
            model = MME4BAT(cdd_strategy=cdd_strategy)

            base_model.fit(rt_x=rt_x, rt_y=rt_y, dt_x=dt_x, dt_y=dt_y)
            model.fit(rt_x=rt_x, rt_y=rt_y, dt_x=dt_x, dt_y=dt_y)

            print(" | MODEL INITIATED")
        else:
            true_prediction_rt = rt_y["run_time_in_seconds"].tolist()
            true_prediction_dt = dt_y["dwell_time_in_seconds"].tolist()

            # getting base model prediction
            base_model_prediction = base_model.predict(rt_x=rt_x, dt_x=dt_x)
            base_model_prediction_rt = base_model_prediction[0][
                "prediction"
            ].tolist()
            base_model_prediction_dt = base_model_prediction[1][
                "prediction"
            ].tolist()

            # getting current model prediction
            model_prediction = model.predict(rt_x=rt_x, dt_x=dt_x)
            model_prediction_rt = model_prediction[0][
                "prediction"
            ].tolist()
            model_prediction_dt = model_prediction[1][
                "prediction"
            ].tolist()

            rt_true_predictions.extend(true_prediction_rt)
            rt_base_model_predictions.extend(base_model_prediction_rt)
            rt_model_predictions.extend(model_prediction_rt)

            dt_true_predictions.extend(true_prediction_dt)
            dt_base_model_predictions.extend(base_model_prediction_dt)
            dt_model_predictions.extend(model_prediction_dt)

            rt_chunk["true_prediction"] = true_prediction_rt
            rt_chunk["base_model_prediction"] = base_model_prediction_rt
            rt_chunk["model_prediction"] = model_prediction_rt

            dt_chunk["true_prediction"] = true_prediction_dt
            dt_chunk["base_model_prediction"] = base_model_prediction_dt
            dt_chunk["model_prediction"] = model_prediction_dt

            result_rt_df = concat([result_rt_df, rt_chunk], ignore_index=True)
            result_dt_df = concat([result_dt_df, dt_chunk], ignore_index=True)

            if active_strategy:
                if (rt_x_buffer is None) or (rt_y_buffer is None):
                    rt_x_buffer = rt_x
                    rt_y_buffer = rt_y
                else:
                    rt_x_buffer = concat([rt_x_buffer, rt_x], ignore_index=True)
                    rt_y_buffer = concat([rt_y_buffer, rt_y], ignore_index=True)

                if (dt_x_buffer is None) or (dt_y_buffer is None):
                    dt_x_buffer = dt_x
                    dt_y_buffer = dt_y
                else:
                    dt_x_buffer = concat([dt_x_buffer, dt_x], ignore_index=True)
                    dt_y_buffer = concat([dt_y_buffer, dt_y], ignore_index=True)

                is_detected = model.is_concept_drift_detected(
                    ni_rt_x=rt_x, ni_rt_y=rt_y, ni_dt_x=dt_x, ni_dt_y=dt_y
                )
                if is_detected:
                    model.incremental_fit(
                        ni_rt_x=rt_x_buffer,
                        ni_rt_y=rt_y_buffer,
                        ni_dt_x=dt_x_buffer,
                        ni_dt_y=dt_y_buffer,
                    )
                    dt_x_buffer = None
                    dt_y_buffer = None
            else:
                model.incremental_fit(
                    ni_rt_x=rt_x, ni_rt_y=rt_y, ni_dt_x=dt_x, ni_dt_y=dt_y
                )

    print("\rDATA STREAMING ENDED.", flush=True)
    print(
        "\rGENERATING & EXPORTING THE RESULTS...",
        end="",
        flush=True,
    )

    cdd_strategy_content = ""
    if active_strategy:
        cdd_strategy_content = f"- {cdd_strategy.__class__.__name__}:\n\n"
        cdd_strategy_content += f"| Attribute | Value |\n|---|---|\n"

        for attr, value in cdd_strategy.get_attributes().items():
            cdd_strategy_content += f"| {attr} | {value} |\n"

    md_file_content = f"""
# Experiment: {label}

## Parameters
- Historical data starting from: {hist_start}
- Historical data ending at: {hist_end}
- Streaming data starting from: {stream_start}
- Streaming data ending at: {stream_end}
- Time interval: {interval_min}
- Concept drift detection strategy: {"ACTIVE" if active_strategy else "PASSIVE"}
- Concept drift detection algorithm: {cdd_strategy.__class__.__name__ if active_strategy else None}
{cdd_strategy_content}

## Results for running time prediction
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | {mean_absolute_error(rt_true_predictions, rt_base_model_predictions)} | {mean_absolute_percentage_error(rt_true_predictions, rt_base_model_predictions) * 100} | {mean_squared_error(rt_true_predictions, rt_base_model_predictions, squared=False)} |
| Base Model with Incremental Learning | {mean_absolute_error(rt_true_predictions, rt_model_predictions)}      | {mean_absolute_percentage_error(rt_true_predictions, rt_model_predictions) * 100}      | {mean_squared_error(rt_true_predictions, rt_model_predictions, squared=False)}      |

## Results for dwell time prediction
### Model Performance Metrics
| Model                               | MAE (s)   | MAPE (%)   | RMSE (s)   |
|-------------------------------------|-----------|------------|------------|
| Base Model (XGBoost)                 | {mean_absolute_error(dt_true_predictions, dt_base_model_predictions)} | {mean_absolute_percentage_error(dt_true_predictions, dt_base_model_predictions) * 100} | {mean_squared_error(dt_true_predictions, dt_base_model_predictions, squared=False)} |
| Base Model with Incremental Learning | {mean_absolute_error(dt_true_predictions, dt_model_predictions)}      | {mean_absolute_percentage_error(dt_true_predictions, dt_model_predictions) * 100}      | {mean_squared_error(dt_true_predictions, dt_model_predictions, squared=False)}      |

    """

    output_dir = os.path.join(
        output_parent_dir,
        f"ex-{label}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/README.md", "w") as f:
        f.write(md_file_content)

    print("\rGENERATING & EXPORTING THE RESULTS ENDED.", flush=True)

    # print(
    #     f"MAE for base model: {mean_absolute_error(true_prediction, base_model_prediction)}"
    # )
    # print(
    #     f"MAPE for base model: {mean_absolute_percentage_error(true_prediction, base_model_prediction) * 100}"
    # )
    # print(
    #     f"RMSE for base model: {mean_squared_error(true_prediction, base_model_prediction, squared=False)}"
    # )
    # print("-----------------------------------------------------------------")
    # print(
    #     f"MAE for incremental online learning model: {mean_absolute_error(true_prediction, model_prediction)}"
    # )
    # print(
    #     f"""MAPE for incremental online learning model: {
    #     mean_absolute_percentage_error(true_prediction, model_prediction) * 100
    # }"""
    # )
    # print(
    #     f"""RMSE for incremental online learning model: {
    #     mean_squared_error(true_prediction, model_prediction, squared=False)
    # }"""
    # )

    # dt_pred_df = dt_df[1000:-100]
    # # dt_pred_df = dt_df[1000:-83]
    # if (
    #     len(dt_pred_df)
    #     == len(true_prediction)
    #     == len(model_prediction)
    #     == len(base_model_prediction)
    # ):
    #     dt_pred_df["true_prediction"] = true_prediction
    #     dt_pred_df["model_prediction"] = model_prediction
    #     dt_pred_df["base_model_prediction"] = base_model_prediction
    # else:
    #     print(
    #         len(true_prediction),
    #         len(model_prediction),
    #         len(base_model_prediction),
    #         dt_pred_df.shape,
    #     )
    #     print("Error: The lengths of the arrays and the DataFrame did not match.")

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

    df_rt = result_rt_df
    df_rt["date"] = to_datetime(df_rt["date"])
    df_rt["start_time"] = to_datetime(df_rt["start_time"], format="%H:%M:%S")
    df_dt = result_dt_df
    df_dt["date"] = to_datetime(df_dt["date"])
    df_dt["arrival_time"] = to_datetime(df_dt["arrival_time"], format="%H:%M:%S")

    directions_rt = df_rt["direction"].unique()
    segments = df_rt["segment"].unique()
    directions_dt = df_dt["direction"].unique()
    bus_stops = df_dt["bus_stop"].unique()

    starting_time = datetime.strptime("06:00:00", "%H:%M:%S")
    ending_time = datetime.strptime("18:00:00", "%H:%M:%S")

    from_time = starting_time

    while from_time < ending_time:
        to_time = from_time + timedelta(minutes=interval_min)
        print(
            f"\rGENERATING & EXPORTING PLOTS FOR EACH SEGMENTS: [{from_time.strftime('%H:%M:%S')} - {to_time.strftime('%H:%M:%S')})",
            end="",
            flush=True,
        )

        for direction in directions_rt:

            for segment in segments:
                filtered_df = df_rt[
                    (df_rt["direction"] == direction)
                    & (df_rt["segment"] == segment)
                    & (df_rt["start_time"].dt.time >= from_time.time())
                    & (df_rt["start_time"].dt.time < to_time.time())
                    ]

                export_at = os.path.join(
                    output_dir,
                    f"rt-d-{direction}-ti-{from_time.strftime('%H-%M-%S')}_{to_time.strftime('%H-%M-%S')}",
                )
                os.makedirs(export_at, exist_ok=True)

                if len(filtered_df) > 0:
                    export_mean_rt_plot_as_image(
                        df=filtered_df,
                        segment=segment,
                        starting_time=from_time,
                        ending_time=to_time,
                        export_at=export_at,
                    )

        from_time = to_time

    print("\rGENERATING & EXPORTING PLOTS FOR EACH SEGMENTS ENDED.", flush=True)

    from_time = starting_time

    while from_time < ending_time:
        to_time = from_time + timedelta(minutes=interval_min)
        print(
            f"\rGENERATING & EXPORTING PLOTS FOR EACH BUS STOP: [{from_time.strftime('%H:%M:%S')} - {to_time.strftime('%H:%M:%S')})",
            end="",
            flush=True,
        )

        for direction in directions_dt:

            for bus_stop in bus_stops:
                filtered_df = df_dt[
                    (df_dt["direction"] == direction)
                    & (df_dt["bus_stop"] == bus_stop)
                    & (df_dt["arrival_time"].dt.time >= from_time.time())
                    & (df_dt["arrival_time"].dt.time < to_time.time())
                    ]

                export_at = os.path.join(
                    output_dir,
                    f"dt-d-{direction}-ti-{from_time.strftime('%H-%M-%S')}_{to_time.strftime('%H-%M-%S')}",
                )
                os.makedirs(export_at, exist_ok=True)

                if len(filtered_df) > 0:
                    export_mean_dt_plot_as_image(
                        df=filtered_df,
                        bus_stop=bus_stop,
                        starting_time=from_time,
                        ending_time=to_time,
                        export_at=export_at,
                    )

        from_time = to_time

    print("\rGENERATING & EXPORTING PLOTS FOR EACH BUS STOP ENDED.", flush=True)


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
    plt.ylabel("Mean dwell time (s)")
    plt.title(
        f"The mean dwell time at bus stop {bus_stop} for each day between {starting_time.time()} and {ending_time.time()}."
    )
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_at}/{bus_stop}.png", dpi=200)
    plt.close()


def export_mean_rt_plot_as_image(
        df: DataFrame,
        segment: int,
        starting_time,
        ending_time,
        export_at: str,
) -> None:
    rt_in_seconds_df = (
        df.groupby("date")["run_time_in_seconds"]
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
        rt_in_seconds_df["run_time_in_seconds"],
        label="True running time",
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
    plt.ylabel("Mean running time (s)")
    plt.title(
        f"The mean running time at segment {segment} for each day between {starting_time.time()} and {ending_time.time()}."
    )
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_at}/{segment}.png", dpi=200)
    plt.close()
