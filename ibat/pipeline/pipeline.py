import os
import warnings
from datetime import datetime, timedelta
from enum import Enum
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


class BatchProcTech(Enum):
    PERIODIC = "Periodic Batch Processing (PBP)"
    EVENT_BASED = "Event-based Batch Processing (EBP)"
    HYBRID = "Hybrid Batch Processing (HBP)"


class CDHandlingStrategy(Enum):
    ACTIVE = "Active"
    PASSIVE = "Passive"


class ModelUpdateStrategy(Enum):
    INCREMENTAL_LEARNING = "Incremental learning"
    SCRATCH_TRAINING = "Retraining from scratch"


class Pipeline:
    def __init__(
        self,
        df: DataFrame,
        hist_start: datetime,
        hist_end: datetime,
        stream_start: datetime,
        stream_end: datetime,
        interval_min: float,
        chunk_size: int,
        cdh_strategy: CDHandlingStrategy = CDHandlingStrategy.PASSIVE,
        cdd_strategy: Optional[IStrategy] = None,
        mu_strategy: ModelUpdateStrategy = ModelUpdateStrategy.INCREMENTAL_LEARNING,
        is_buffer_enabled: Optional[bool] = False,
        output_parent_dir: Optional[str] = "./",
        label: Optional[str] = "",
    ) -> None:
        self._df = df
        self._hist_start = hist_start
        self._hist_end = hist_end
        self._stream_start = stream_start
        self._stream_end = stream_end

        self._interval_min = interval_min
        self._chunk_size = chunk_size
        self._bp_technique = self.init_bp_technique()

        if (cdh_strategy == CDHandlingStrategy.ACTIVE) and cdd_strategy is None:
            raise ValueError(
                "cdd_strategy must be provided when cdh_strategy is ACTIVE."
            )
        self._cdh_strategy = cdh_strategy
        self._cdd_strategy = cdd_strategy

        self._mu_strategy = mu_strategy
        self._is_buffer_enabled = is_buffer_enabled
        self._output_parent_dir = output_parent_dir
        self._label = label

    def init_bp_technique(self) -> BatchProcTech:
        interval_min, chunk_size = self._interval_min, self._chunk_size

        if interval_min == 0 and chunk_size == 0:
            raise ValueError("Both interval_min & chunk_size cannot be set to zero.")
        else:
            count_not_enough = False
            is_end_reached = False

            hybrid_bp = interval_min != 0 and chunk_size != 0
            scheduled_bp = interval_min != 0 and chunk_size == 0

            if hybrid_bp:
                return BatchProcTech.HYBRID
            elif scheduled_bp:
                return BatchProcTech.PERIODIC
            else:
                return BatchProcTech.EVENT_BASED

    def run_exp(self) -> None:
        print(
            f"BATCH PROCESSING TECHNIQUE: {self._bp_technique} | "
            f"CONCEPT DRIFT HANDLING STRATEGY: {self._cdh_strategy} | "
            f"STRATEGY TO UPDATE THE OUTDATED MODEL: {self._mu_strategy}"
        )

        df = self._df.copy()

        df["datetime"] = to_datetime(
            df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
        )
        df.sort_values(by="datetime", inplace=True)
