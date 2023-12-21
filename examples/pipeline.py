from datetime import datetime
from src.pipeline import run_exp
from src.concept_drift_detector.strategies import ADWIN
from src.concept_drift_detector.strategies import DDM
from src.concept_drift_detector.strategies import PageHinkley

def datetime_from_string(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, '%Y-%m-%d')


if __name__ == "__main__":
    historical_data_starting_from = datetime_from_string("2021-10-01")
    historical_data_ending_at = datetime_from_string("2021-10-10")
    streaming_data_starting_from = datetime_from_string("2021-10-10")
    streaming_data_ending_at = datetime_from_string("2022-11-01")
    time_interval = 60
    isActive = True
    # cdd_strategy = ADWIN(delta=0.2)
    # cdd_strategy = PageHinkley(threshold=6500)
    cdd_strategy = DDM(
                warning_level=0.5,
                drift_level=1,
                min_num_instances=5,
            )
    folder_path_to_save_result = "../experiments"
    experiment_label = "m-xgb-s-xgb_model"

    run_exp(
        hist_start=historical_data_starting_from,
        hist_end=historical_data_ending_at,
        stream_start=streaming_data_starting_from,
        stream_end=streaming_data_ending_at,
        interval_min=time_interval,
        isActive=isActive,
        cdd_strategy=cdd_strategy,
        output_parent_dir=folder_path_to_save_result,
        label=experiment_label,
    )
