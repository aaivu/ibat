from datetime import datetime
from src.pipeline import run_exp


def datetime_from_string(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, '%Y-%m-%d')


if __name__ == "__main__":
    historical_data_starting_from = datetime_from_string("2021-10-01")
    historical_data_ending_at = datetime_from_string("2021-10-10")
    streaming_data_starting_from = datetime_from_string("2021-10-10")
    streaming_data_ending_at = datetime_from_string("2022-11-01")
    time_interval = 60
    isActive = 'True'
    folder_path_to_save_result = "../experiments"
    experiment_label = "m-xgb-s-xgb_model"

    run_exp(
        hist_start=historical_data_starting_from,
        hist_end=historical_data_ending_at,
        stream_start=streaming_data_starting_from,
        stream_end=streaming_data_ending_at,
        interval_min=time_interval,
        isActive=isActive,
        output_parent_dir=folder_path_to_save_result,
        label=experiment_label,
    )
