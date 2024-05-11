from datetime import datetime

from src.concept_drift_detector.strategies import DDM
from src.datasets import BUS_654_FEATURES_ENCODED_DWELL_TIMES
from src.pipeline import run_dt_exp


def datetime_from_string(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%d")


if __name__ == "__main__":
    dwell_time_df = BUS_654_FEATURES_ENCODED_DWELL_TIMES.dataframe
    historical_data_starting_from = datetime_from_string("2021-10-01")
    historical_data_ending_at = datetime_from_string("2022-02-01")
    streaming_data_starting_from = datetime_from_string("2022-02-01")
    streaming_data_ending_at = datetime_from_string("2022-11-01")
    time_interval = 60 * 2
    chunk_size = 100
    active_strategy = True
    is_buffer_enabled = False
    cdd_strategy = DDM(
        warning_level=0.1,
        drift_level=1.5,
        min_num_instances=1,
    )
    incremental_learning = True
    folder_path_to_save_result = "../experiments"
    experiment_label = "m-xgb-s-xgb_model-d-batch_r2"

    run_dt_exp(
        dt_df=dwell_time_df,
        hist_start=historical_data_starting_from,
        hist_end=historical_data_ending_at,
        stream_start=streaming_data_starting_from,
        stream_end=streaming_data_ending_at,
        interval_min=time_interval,
        chunk_size=chunk_size,
        active_strategy=active_strategy,
        is_buffer_enabled=is_buffer_enabled,
        cdd_strategy=cdd_strategy,
        incremental_learning=incremental_learning,
        output_parent_dir=folder_path_to_save_result,
        label=experiment_label,
    )
