import numpy as np
import pandas as pd
from ibat.concept_drift_detector import CDD
from ibat.concept_drift_detector.strategies import PageHinkley


sorted_mean_arrival_time = pd.read_csv("../../asserts/datasets/input.csv")

stream = np.array(list(sorted_mean_arrival_time["arrival_time_in_seconds"]))

strategy = PageHinkley(threshold=6500)

cdd = CDD(strategy=strategy)

is_detected = cdd.is_concept_drift_detected(None, stream, None)

print(is_detected)
