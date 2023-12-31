from typing import Any, Dict

from sklearn.metrics import mean_absolute_percentage_error
from frouros.detectors.concept_drift import DDM as fDDM, DDMConfig
from frouros.metrics import PrequentialError
from src.concept_drift_detector.strategies.istrategy import IStrategy


class DDM(IStrategy):
    def __init__(
        self,
        warning_level: float,
        drift_level: float,
        min_num_instances: int,
    ) -> None:
        self._warning_level = warning_level
        self._drift_level = drift_level
        self._min_num_instances = min_num_instances

        self._ddm_config = DDMConfig(
            warning_level=warning_level,
            drift_level=drift_level,
            min_num_instances=min_num_instances,
        )
        self._ddm = fDDM(config=self._ddm_config)
        self._metric = PrequentialError(alpha=1.0)

    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        is_detected = False
        idx_drift, idx_warning = [], []
        metric_error = 0

        for (i, x), (j, y) in zip(ni_x.iterrows(), ni_y.iterrows()):
            x = x.to_frame().transpose()
            y = y.to_frame().transpose()
            y_pred = model.predict(x)

            mape = mean_absolute_percentage_error(y, y_pred)
            metric_error = self._metric(error_value=mape)

            self._ddm.update(value=mape)
            status = self._ddm.status

            if status["drift"] and not is_detected:
                is_detected = True
                idx_drift.append(i)
                # print(
                #     f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}"
                # )
            if status["warning"]:
                # print(
                #     f"warning detected: {i} MAPE={mape:.4f} Accuracy : {1 - metric_error:.4f} "
                # )
                idx_warning.append(i)
        # if not is_detected:
        #     print("No concept drift detected")

        # print(f"Final accuracy: {1 - metric_error:.4f}\n")
        # print("warning index : ", idx_warning)
        # print("drift index : ", idx_drift)

        return is_detected

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "Warning Level Factor": float(self._warning_level),
            "Drift Level Factor": float(self._drift_level),
            "Minimum Numbers of Instances to Start Looking for Changes": self._min_num_instances,
        }
