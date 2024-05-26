from copy import deepcopy
from typing import Optional

from ibat.concept_drift_detector import CDD
from ibat.concept_drift_detector.strategies import IStrategy
from ibat.models.base_models.ensemble.xgboost import XGBRegressor


class MME4BRT:
    def __init__(self, cdd_strategy: Optional[IStrategy] = None) -> None:
        self.xgb_rt_regressor = XGBRegressor()

        if cdd_strategy:
            self._cdd_of_xgb_rt_regressor = CDD(
                strategy=deepcopy(cdd_strategy),
            )

    def fit(self, rt_x, rt_y) -> None:
        self.xgb_rt_regressor.fit(rt_x, rt_y)

    def incremental_fit(self, ni_rt_x, ni_rt_y) -> None:
        self.xgb_rt_regressor.incremental_fit(ni_rt_x, ni_rt_y)

    def predict(self, rt_x):
        rt_y = self.xgb_rt_regressor.predict(rt_x)
        rt_y.loc[rt_y["prediction"] < 0, "prediction"] = 0

        return rt_y

    def is_concept_drift_detected(self, ni_rt_x, ni_rt_y) -> bool:
        try:
            is_detected_1 = self._cdd_of_xgb_rt_regressor.is_concept_drift_detected(
                model=self.xgb_rt_regressor,
                ni_x=ni_rt_x,
                ni_y=ni_rt_y,
            )
            print(
                f" | CDD at xgb_rt_regressor: {is_detected_1}"
            )
            return is_detected_1
        except NameError as e:
            raise e
