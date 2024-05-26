import os
import warnings
from datetime import datetime, timedelta
from time import time
from typing import Optional

import matplotlib.pyplot as plt
from numpy import mean
from pandas import concat, DataFrame, to_datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from ibat.concept_drift_detector.strategies import IStrategy
from ibat.models.use_cases.running_time.bus import MME4BRT


def run_rt_exp(
    rt_df: DataFrame,
    hist_start: datetime,
    hist_end: datetime,
    stream_start: datetime,
    stream_end: datetime,
    interval_min: float,
    chunk_size: int,
    active_strategy: Optional[bool] = False,
    cdd_strategy: Optional[IStrategy] = None,
    incremental_learning: Optional[bool] = True,
    is_buffer_enabled: Optional[bool] = False,
    output_parent_dir: Optional[str] = "./",
    label: Optional[str] = "",
) -> None:
    """
    Run a running time experiment.

    Args:
        rt_df: running time dataframe to contact the experiment.
        hist_start: Start timestamp (inclusive) for historical data.
        hist_end: End timestamp (exclusive) for historical data.
        stream_start: Start timestamp (inclusive) for streaming data.
        stream_end: End timestamp (exclusive) for streaming data.
        interval_min: Minimum waiting time (in minutes) for data processing.
        chunk_size: Minimum required amount of data for data processing.
        active_strategy: Flag indicating whether to use active strategy (default is False).
        cdd_strategy: Strategy to be used for detecting concept drift. Required if active_strategy is True.
        incremental_learning: Flag indicating whether to perform incremental learning (default is True).
        is_buffer_enabled: Flag indicating whether buffering is enabled (default is False).
        output_parent_dir: Parent directory's path to save experiment results.
        label: Experiment label (default is an empty string).

    Returns:
        None
    """
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

    if incremental_learning:
        mu_strategy = "INCREMENTAL LEARNING"
    else:
        mu_strategy = "RETRAINING FROM SCRATCH"

    print(
        f"BATCH PROCESSING TECHNIQUE: {bp_technique} | CONCEPT DRIFT HANDLING STRATEGY: {strategy} | STRATEGY TO UPDATE THE OUTDATED MODEL: {mu_strategy}"
    )

    rt_df["start_datetime"] = to_datetime(
        rt_df["date"] + " " + rt_df["start_time"], format="%Y-%m-%d %H:%M:%S"
    )
    rt_df.sort_values(by="start_datetime", inplace=True)

    base_model: Optional[MME4BRT] = None
    model: Optional[MME4BRT] = None

    result_rt_df = DataFrame(
        columns=rt_df.columns.tolist()
        + ["true_prediction", "base_model_prediction", "model_prediction"]
    )

    true_predictions = []
    base_model_predictions = []
    model_predictions = []

    processing_times = []

    rt_x_buffer: Optional[DataFrame] = None
    rt_y_buffer: Optional[DataFrame] = None

    from_date_time = hist_start
    to_date_time = hist_end

    while from_date_time < stream_end:
        if hybrid_bp and count_not_enough:
            temp_df = rt_df.loc[
                (rt_df["start_datetime"] >= from_date_time),
                :,
            ].reset_index(drop=True)
            if temp_df.shape[0] < chunk_size:
                is_end_reached = True
            else:
                to_date_time = temp_df["start_datetime"].iloc[
                    chunk_size - 1
                ] + timedelta(minutes=1)

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
        count_not_enough = rt_chunk.shape[0] < chunk_size

        rt_full: DataFrame = rt_df.loc[
            rt_df["start_datetime"] < to_date_time,
            :,
        ].reset_index(drop=True)

        if scheduled_bp or (hybrid_bp and not count_not_enough) or is_end_reached:
            from_date_time = (
                stream_start if from_date_time == hist_start else to_date_time
            )
            to_date_time = from_date_time + timedelta(minutes=interval_min)

            if len(rt_chunk) == 0:
                print(" | NO INSTANCES")
                continue
            else:
                print(f" | NUMBER OF INSTANCES: {len(rt_chunk):04d}", end="")

            numeric_rt_chunk = rt_chunk.select_dtypes(include="number")

            rt_x: DataFrame = numeric_rt_chunk.drop(columns=["run_time_in_seconds"])
            rt_y: DataFrame = numeric_rt_chunk[["run_time_in_seconds"]]

            numeric_rt_full = rt_full.select_dtypes(include="number")

            rt_full_x: DataFrame = numeric_rt_full.drop(columns=["run_time_in_seconds"])
            rt_full_y: DataFrame = numeric_rt_full[["run_time_in_seconds"]]

            if not model:
                base_model = MME4BRT()
                model = MME4BRT(cdd_strategy=cdd_strategy)

                base_model.fit(rt_x=rt_x, rt_y=rt_y)
                model.fit(rt_x=rt_x, rt_y=rt_y)

                print(" | MODEL INITIATED")
            else:
                true_prediction = rt_y["run_time_in_seconds"].tolist()
                base_model_prediction = base_model.predict(rt_x=rt_x)[
                    "prediction"
                ].tolist()
                model_prediction = model.predict(rt_x=rt_x)["prediction"].tolist()

                true_predictions.extend(true_prediction)
                base_model_predictions.extend(base_model_prediction)
                model_predictions.extend(model_prediction)

                rt_chunk["true_prediction"] = true_prediction
                rt_chunk["base_model_prediction"] = base_model_prediction
                rt_chunk["model_prediction"] = model_prediction

                result_rt_df = concat([result_rt_df, rt_chunk], ignore_index=True)

                start_time = time()

                if active_strategy:
                    if is_buffer_enabled:
                        if (rt_x_buffer is None) or (rt_y_buffer is None):
                            rt_x_buffer = rt_x
                            rt_y_buffer = rt_y
                        else:
                            rt_x_buffer = concat([rt_x_buffer, rt_x], ignore_index=True)
                            rt_y_buffer = concat([rt_y_buffer, rt_y], ignore_index=True)
                        ni_rt_x = rt_x_buffer
                        ni_rt_y = rt_y_buffer
                    else:
                        ni_rt_x = rt_x
                        ni_rt_y = rt_y

                    is_detected = model.is_concept_drift_detected(
                        ni_rt_x=rt_x, ni_rt_y=rt_y
                    )
                    if is_detected:
                        if incremental_learning:
                            model.incremental_fit(
                                ni_rt_x=ni_rt_x,
                                ni_rt_y=ni_rt_y,
                            )
                        else:
                            model = MME4BRT(cdd_strategy=cdd_strategy)
                            model.fit(
                                rt_x=rt_full_x,
                                rt_y=rt_full_y,
                            )
                        rt_x_buffer = None
                        rt_y_buffer = None
                else:
                    if incremental_learning:
                        model.incremental_fit(ni_rt_x=rt_x, ni_rt_y=rt_y)
                    else:
                        model = MME4BRT(cdd_strategy=cdd_strategy)
                        model.fit(
                            rt_x=rt_full_x,
                            rt_y=rt_full_y,
                        )
                    print()

                end_time = time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
        else:
            print(
                f" | NUMBER OF INSTANCES: {len(rt_chunk):04d} | COUNT IS NOT ENOUGH. WAITING FOR MORE DATA POINTS."
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

    base_model_mae = mean_absolute_error(true_predictions, base_model_predictions)
    model_mae = mean_absolute_error(true_predictions, model_predictions)

    base_model_rmse = root_mean_squared_error(true_predictions, base_model_predictions)
    model_rmse = root_mean_squared_error(true_predictions, model_predictions)

    base_model_mape = mean_absolute_percentage_error(
        true_predictions, base_model_predictions
    )
    model_mape = mean_absolute_percentage_error(true_predictions, model_predictions)

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
- Concept drift detection algorithm: {cdd_strategy.__class__.__name__ if active_strategy else None}
- Strategy to update the outdated model: {mu_strategy}
- Is buffer enabled: {is_buffer_enabled}
{cdd_strategy_content}

## Results
- Model performance metrics:

| Model                                     | MAE (s)              | RMSE (s)              | MAPE (%)              |
|-------------------------------------------|----------------------|-----------------------|-----------------------|
| Base model without concept drift handling | {base_model_mae:.3f} | {base_model_rmse:.3f} | {base_model_mape:.3f} |
| Base model with concept drift handling    | {model_mae:.3f}      | {model_rmse:.3f}      | {model_mape:.3f}      |

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

    df = result_rt_df
    df["date"] = to_datetime(df["date"])
    df["start_time"] = to_datetime(df["start_time"], format="%H:%M:%S")

    directions = df["direction"].unique()
    segments = df["segment"].unique()

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
            for segment in segments:
                filtered_df = df[
                    (df["direction"] == direction)
                    & (df["segment"] == segment)
                    & (df["start_time"].dt.time >= from_time.time())
                    & (df["start_time"].dt.time < to_time.time())
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

    print(
        "\rGENERATING & EXPORTING PLOTS FOR EACH BUS STOP ENDED.               ",
        flush=True,
    )


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
        label="Actual running time",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        x,
        base_model_prediction_df["base_model_prediction"],
        label="Prediction of base model",
        marker="o",
        linestyle="--",
        color="green",
    )
    plt.plot(
        x,
        model_prediction_df["model_prediction"],
        label="Prediction of base model with experimental setup",
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
