import os
import warnings
from datetime import datetime, timedelta
from time import time
from typing import Optional

import matplotlib.pyplot as plt
from numpy import mean
from pandas import concat, DataFrame, to_datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from src.concept_drift_detector.strategies import IStrategy
from src.datasets import (
    BUS_654_FEATURES_ADDED_RUNNING_TIMES,
    BUS_654_FEATURES_ENCODED_DWELL_TIMES,
)
from src.models.use_cases.arrival_time.bus import MME4BAT


def run_exp(
    dt_df: DataFrame,
    hist_start: datetime,
    hist_end: datetime,
    stream_start: datetime,
    stream_end: datetime,
    interval_min: float,
    chunk_size: int,
    active_strategy: Optional[bool] = False,
    is_buffer_enabled: Optional[bool] = False,
    cdd_strategy: Optional[IStrategy] = None,
    incremental_learning: Optional[bool] = True,
    output_parent_dir: Optional[str] = "./",
    label: Optional[str] = "",
) -> None:
    """
    Run the experiment.

    Args:
        dt_df: Dwell time dataframe to contact the experiment.
        hist_start: Start timestamp (inclusive) for historical data.
        hist_end: End timestamp (exclusive) for historical data.
        stream_start: Start timestamp (inclusive) for streaming data.
        stream_end: End timestamp (exclusive) for streaming data.
        interval_min: Minimum waiting time (in minutes) for data processing.
        chunk_size: Minimum required amount of data for data processing.
        active_strategy: Flag indicating whether to use active strategy (default is False).
        is_buffer_enabled: To do (default is False).
        cdd_strategy: Strategy to be used for detecting concept drift. Required if active_strategy is True.
        incremental_learning: To do (default is True).
        output_parent_dir: Parent directory's path to save experiment results.
        label: Experiment label (default is an empty string).

    Returns:
        None
    """
    # To do: warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    warnings.filterwarnings("ignore")

    if interval_min == 0 and chunk_size == 0:
        raise ValueError("Both interval_min & chunk_size cannot be set to zero.")
    else:
        count_not_enough = False
        is_end_reached = False

        hybrid_bp = interval_min != 0 and chunk_size != 0
        scheduled_bp = interval_min != 0 and chunk_size == 0

        if hybrid_bp:
            bp_technique = "HYBRID"
        elif scheduled_bp:
            bp_technique = "SCHEDULED"
        else:
            bp_technique = "PERIODIC"

    if active_strategy and cdd_strategy is None:
        raise ValueError("cdd_strategy must be provided when active_strategy is True.")
    else:
        strategy = "ACTIVE" if active_strategy else "PASSIVE"

    print(f"BATCH PROCESSING TECHNIQUE: {bp_technique} | CONCEPT DRIFT HANDLING STRATEGY: {strategy}")

    rt_df: DataFrame = BUS_654_FEATURES_ADDED_RUNNING_TIMES.dataframe
    # dt_df: DataFrame = BUS_654_FEATURES_ENCODED_DWELL_TIMES.dataframe

    dt_df["arrival_datetime"] = to_datetime(
        dt_df["date"] + " " + dt_df["arrival_time"], format="%Y-%m-%d %H:%M:%S"
    )
    dt_df.sort_values(by="arrival_datetime", inplace=True)

    base_model: Optional[MME4BAT] = None
    model: Optional[MME4BAT] = None

    result_dt_df = DataFrame(
        columns=dt_df.columns.tolist()
        + ["true_prediction", "base_model_prediction", "model_prediction"]
    )

    true_predictions = []
    base_model_predictions = []
    model_predictions = []

    processing_times = []

    dt_x_buffer: Optional[DataFrame] = None
    dt_y_buffer: Optional[DataFrame] = None

    from_date_time = hist_start
    to_date_time = hist_end

    while from_date_time < stream_end:
        if hybrid_bp and count_not_enough:
            temp_df = dt_df.loc[
                (dt_df["arrival_datetime"] >= from_date_time),
                :,
            ].reset_index(drop=True)
            if temp_df.shape[0] < chunk_size:
                is_end_reached = True
            else:
                to_date_time = temp_df["arrival_datetime"].iloc[chunk_size - 1] + timedelta(minutes=1)

        print(
            f"DATA STREAM: [{from_date_time.strftime('%Y-%m-%d %H:%M:%S')} - {to_date_time.strftime('%Y-%m-%d %H:%M:%S')})",
            end="",
            flush=True,
        )

        dt_chunk: DataFrame = dt_df.loc[
            (from_date_time <= dt_df["arrival_datetime"])
            & (dt_df["arrival_datetime"] < to_date_time),
            :,
        ].reset_index(drop=True)
        count_not_enough = dt_chunk.shape[0] < chunk_size

        if (
            scheduled_bp
            or (hybrid_bp and not count_not_enough)
            or is_end_reached
        ):
            from_date_time = (
                stream_start if from_date_time == hist_start else to_date_time
            )
            to_date_time = from_date_time + timedelta(minutes=interval_min)

            if len(dt_chunk) == 0:
                print(" | NO INSTANCES")
                continue
            else:
                print(f" | NUMBER OF INSTANCES: {len(dt_chunk):04d}", end="")

            numeric_dt_chunk = dt_chunk.select_dtypes(include="number")

            dt_x: DataFrame = numeric_dt_chunk.drop(columns=["dwell_time_in_seconds"])
            dt_y: DataFrame = numeric_dt_chunk[["dwell_time_in_seconds"]]

            if not model:
                base_model = MME4BAT()
                model = MME4BAT(cdd_strategy=cdd_strategy)

                base_model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)
                model.fit(rt_x=None, rt_y=None, dt_x=dt_x, dt_y=dt_y)

                print(" | MODEL INITIATED")
            else:
                true_prediction = dt_y["dwell_time_in_seconds"].tolist()
                base_model_prediction = base_model.predict(rt_x=None, dt_x=dt_x)[
                    "prediction"
                ].tolist()
                model_prediction = model.predict(rt_x=None, dt_x=dt_x)[
                    "prediction"
                ].tolist()

                true_predictions.extend(true_prediction)
                base_model_predictions.extend(base_model_prediction)
                model_predictions.extend(model_prediction)

                dt_chunk["true_prediction"] = true_prediction
                dt_chunk["base_model_prediction"] = base_model_prediction
                dt_chunk["model_prediction"] = model_prediction

                result_dt_df = concat([result_dt_df, dt_chunk], ignore_index=True)

                start_time = time()

                if active_strategy:
                    if is_buffer_enabled:
                        if (dt_x_buffer is None) or (dt_y_buffer is None):
                            dt_x_buffer = dt_x
                            dt_y_buffer = dt_y
                        else:
                            dt_x_buffer = concat([dt_x_buffer, dt_x], ignore_index=True)
                            dt_y_buffer = concat([dt_y_buffer, dt_y], ignore_index=True)
                        ni_dt_x = dt_x_buffer
                        ni_dt_y = dt_y_buffer
                    else:
                        ni_dt_x = dt_x
                        ni_dt_y = dt_y

                    is_detected = model.is_concept_drift_detected(
                        ni_rt_x=None, ni_rt_y=None, ni_dt_x=dt_x, ni_dt_y=dt_y
                    )
                    if is_detected:
                        model.incremental_fit(
                            ni_rt_x=None,
                            ni_rt_y=None,
                            ni_dt_x=ni_dt_x,
                            ni_dt_y=ni_dt_y,
                        )
                        dt_x_buffer = None
                        dt_y_buffer = None
                else:
                    model.incremental_fit(
                        ni_rt_x=None, ni_rt_y=None, ni_dt_x=dt_x, ni_dt_y=dt_y
                    )
                    print()

                end_time = time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
        else:
            print(f" | NUMBER OF INSTANCES: {len(dt_chunk):04d} | COUNT IS NOT ENOUGH. WAITING FOR MORE DATA POINTS.")

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

    base_model_mae = mean_absolute_error(true_predictions, base_model_predictions)
    model_mae = mean_absolute_error(true_predictions, model_predictions)

    base_model_rmse = root_mean_squared_error(true_predictions, base_model_predictions)
    model_rmse = root_mean_squared_error(true_predictions, model_predictions)

    md_file_content = f"""
# Experiment: {label}

## Parameters
- Historical data starting from: {hist_start}
- Historical data ending at: {hist_end}
- Streaming data starting from: {stream_start}
- Streaming data ending at: {stream_end}
- Minimum waiting time: {interval_min} minutes
- Minimum required amount of data: {chunk_size}
- Batch processing technique: {bp_technique}
- Concept drift handling strategy: {strategy}
- Is buffer enabled: {is_buffer_enabled}
- Concept drift detection algorithm: {cdd_strategy.__class__.__name__ if active_strategy else None}
{cdd_strategy_content}

## Results
- Model performance metrics:

| Model                                | MAE (s)              | RMSE (s)              |
|--------------------------------------|----------------------|-----------------------|
| Base model (XGBoost)                 | {base_model_mae:.3f} | {base_model_rmse:.3f} |
| Base model with incremental learning | {model_mae:.3f}      | {model_rmse:.3f}      |

- Error reduction percentage in terms of MAE: {(base_model_mae - model_mae) * 100 / base_model_mae:.3f} %
- Average processing time after the batch preparation: {mean(processing_times) * 1000:.3f} ms
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

    df = result_dt_df
    df["date"] = to_datetime(df["date"])
    df["arrival_time"] = to_datetime(df["arrival_time"], format="%H:%M:%S")

    directions = df["direction"].unique()
    bus_stops = df["bus_stop"].unique()

    starting_time = datetime.strptime("06:00:00", "%H:%M:%S")
    ending_time = datetime.strptime("18:00:00", "%H:%M:%S")

    from_time = starting_time

    while from_time < ending_time:
        to_time = from_time + timedelta(minutes=60)
        print(
            f"\rGENERATING & EXPORTING PLOTS FOR EACH BUS STOP: [{from_time.strftime('%H:%M:%S')} - {to_time.strftime('%H:%M:%S')})",
            end="",
            flush=True,
        )

        for direction in directions:
            for bus_stop in bus_stops:
                filtered_df = df[
                    (df["direction"] == direction)
                    & (df["bus_stop"] == bus_stop)
                    & (df["arrival_time"].dt.time >= from_time.time())
                    & (df["arrival_time"].dt.time < to_time.time())
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
