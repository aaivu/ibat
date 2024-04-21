from datetime import datetime
from src.pipeline import run_exp
from src.concept_drift_detector.strategies import DDM


def datetime_from_string(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%d")


if __name__ == "__main__":
    historical_data_starting_from = datetime_from_string("2021-10-01")
    historical_data_ending_at = datetime_from_string("2022-01-01")
    streaming_data_starting_from = datetime_from_string("2022-01-01")
    streaming_data_ending_at = datetime_from_string("2022-08-01")
    time_interval = 60
    chunk_size = 100
    active_strategy = True
    cdd_strategy = DDM(
        warning_level=0.1,
        drift_level=3,
        min_num_instances=2,
    )
    folder_path_to_save_result = "../experiments"
    experiment_label = "m-xgb-s-xgb_model-d-batch_r2"

    run_exp(
        hist_start=historical_data_starting_from,
        hist_end=historical_data_ending_at,
        stream_start=streaming_data_starting_from,
        stream_end=streaming_data_ending_at,
        interval_min=time_interval,
        chunk_size=chunk_size,
        active_strategy=active_strategy,
        cdd_strategy=cdd_strategy,
        output_parent_dir=folder_path_to_save_result,
        label=experiment_label,
    )
