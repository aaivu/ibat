from copy import deepcopy
from typing import Optional

from src.concept_drift_detector import CDD
from src.concept_drift_detector.strategies import IStrategy
from src.models._base_models.xgboost import XGBClassifier, XGBRegressor


class MME4BAT:
    def __init__(self, cdd_strategy: Optional[IStrategy] = None) -> None:
        self.xgb_rt_regressor = XGBRegressor()
        self.xgb_dt_classifier = XGBClassifier()
        self.xgb_dt_regressor = XGBRegressor()

        if cdd_strategy:
            self._cdd_of_xgb_rt_regressor = CDD(
                strategy=deepcopy(cdd_strategy),
            )
            self._cdd_of_xgb_dt_classifier = CDD(
                strategy=deepcopy(cdd_strategy),
            )
            self._cdd_of_xgb_dt_regressor = CDD(
                strategy=deepcopy(cdd_strategy),
            )

    def fit(self, rt_x, rt_y, dt_x, dt_y) -> None:
        rt_x_gt_0 = rt_x[rt_y.iloc[:, 0] > 0]
        rt_y_gt_0 = rt_y[rt_y.iloc[:, 0] > 0]
        self.xgb_rt_regressor.fit(rt_x_gt_0, rt_y_gt_0)

        is_dt_gt_0 = dt_y.iloc[:, 0].apply(lambda dt: 1 if dt > 0 else 0)
        self.xgb_dt_classifier.fit(dt_x, is_dt_gt_0)

        dt_x_gt_0 = dt_x[dt_y.iloc[:, 0] > 0]
        dt_y_gt_0 = dt_y[dt_y.iloc[:, 0] > 0]
        self.xgb_dt_regressor.fit(dt_x_gt_0, dt_y_gt_0)

    def incremental_fit(self, ni_rt_x, ni_rt_y, ni_dt_x, ni_dt_y) -> None:
        ni_rt_x_gt_0 = ni_rt_x[ni_rt_y.iloc[:, 0] > 0]
        ni_rt_y_gt_0 = ni_rt_y[ni_rt_y.iloc[:, 0] > 0]
        self.xgb_rt_regressor.incremental_fit(ni_rt_x_gt_0, ni_rt_y_gt_0)

        is_ni_dt_gt_0 = ni_dt_y.iloc[:, 0].apply(lambda dt: 1 if dt > 0 else 0)
        self.xgb_dt_classifier.incremental_fit(ni_dt_x, is_ni_dt_gt_0)

        ni_dt_x_gt_0 = ni_dt_x[ni_dt_y.iloc[:, 0] > 0]
        ni_dt_y_gt_0 = ni_dt_y[ni_dt_y.iloc[:, 0] > 0]
        self.xgb_dt_regressor.incremental_fit(ni_dt_x_gt_0, ni_dt_y_gt_0)

    def predict(self, rt_x, dt_x):
        rt_y = self.xgb_rt_regressor.predict(rt_x)

        is_dt_gt_0 = self.xgb_dt_classifier.predict(dt_x)

        dt_x_gt_0 = dt_x[is_dt_gt_0.iloc[:, 0] == 1]
        dt_y_gt_0 = self.xgb_dt_regressor.predict(dt_x_gt_0)

        dt_y = is_dt_gt_0.copy()
        dt_y.loc[dt_y["prediction"] > 0, "prediction"] = dt_y_gt_0[
            "prediction"
        ].to_numpy()
        dt_y.loc[dt_y["prediction"] < 0, "prediction"] = 0

        return [rt_y, dt_y]

    def is_concept_drift_detected(self, ni_rt_x, ni_rt_y, ni_dt_x, ni_dt_y) -> bool:
        try:
            is_detected_1 = self._cdd_of_xgb_rt_regressor.is_concept_drift_detected(
                model=self.xgb_rt_regressor,
                ni_x=ni_rt_x,
                ni_y=ni_rt_y,
            )

            is_detected_2 = self._cdd_of_xgb_dt_classifier.is_concept_drift_detected(
                model=self.xgb_dt_classifier,
                ni_x=ni_dt_x,
                ni_y=ni_dt_y,
            )

            ni_dt_x_gt_0 = ni_dt_x[ni_dt_y.iloc[:, 0] > 0]
            ni_dt_y_gt_0 = ni_dt_y[ni_dt_y.iloc[:, 0] > 0]

            is_detected_3 = self._cdd_of_xgb_dt_regressor.is_concept_drift_detected(
                model=self.xgb_dt_regressor,
                ni_x=ni_dt_x_gt_0,
                ni_y=ni_dt_y_gt_0,
            )
            print(
                f" | CDD at xgb_rt_regressor: {is_detected_1} | CDD at xgb_dt_classifier: {is_detected_2} | CDD at xgb_dt_regressor: {is_detected_3}"
            )
            return is_detected_1 or is_detected_2 or is_detected_3
        except NameError as e:
            raise e
