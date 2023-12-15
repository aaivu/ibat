from src.models._base_models.xgboost import XGBClassifier, XGBRegressor


class MME4BAT:
    def __init__(self):
        # self.xgb_rt_regressor = XGBRegressor()

        self.xgb_dt_classifier = XGBClassifier()
        self.xgb_dt_regressor = XGBRegressor()

    def fit(self, rt_x, rt_y, dt_x, dt_y) -> None:
        # self.xgb_rt_regressor.fit(rt_x, rt_y)

        is_dt_gt_0 = dt_y.iloc[:, 0].apply(lambda dt: 1 if dt > 0 else 0)
        self.xgb_dt_classifier.fit(dt_x, is_dt_gt_0)

        dt_x_gt_0 = dt_x[dt_y.iloc[:, 0] > 0]
        dt_y_gt_0 = dt_y[dt_y.iloc[:, 0] > 0]
        self.xgb_dt_regressor.fit(dt_x_gt_0, dt_y_gt_0)

    def incremental_fit(self, ni_rt_x, ni_rt_y, ni_dt_x, ni_dt_y) -> None:
        # self.xgb_rt_regressor.incremental_fit(ni_rt_x, ni_rt_y)

        is_ni_dt_gt_0 = ni_dt_y.iloc[:, 0].apply(lambda dt: 1 if dt > 0 else 0)
        self.xgb_dt_classifier.incremental_fit(ni_dt_x, is_ni_dt_gt_0)

        ni_dt_x_gt_0 = ni_dt_x[ni_dt_y.iloc[:, 0] > 0]
        ni_dt_y_gt_0 = ni_dt_y[ni_dt_y.iloc[:, 0] > 0]
        self.xgb_dt_regressor.incremental_fit(ni_dt_x_gt_0, ni_dt_y_gt_0)

    def predict(self, rt_x, dt_x):
        # rt_y = self.xgb_rt_regressor.predict(rt_x)

        is_dt_gt_0 = self.xgb_dt_classifier.predict(dt_x)

        dt_x_gt_0 = dt_x[is_dt_gt_0.iloc[:, 0] == 1]
        dt_y_gt_0 = self.xgb_dt_regressor.predict(dt_x_gt_0)

        dt_y = is_dt_gt_0.copy()
        dt_y.loc[dt_y["prediction"] > 0, "prediction"] = dt_y_gt_0[
            "prediction"
        ].to_numpy()
        dt_y.loc[dt_y["prediction"] < 0, "prediction"] = 0

        return dt_y
