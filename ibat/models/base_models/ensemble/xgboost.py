from ibat.models.base_models.base_models import XGBoost


class XGBClassifier(XGBoost):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {
            "objective": "binary:hinge",
            # "eval_metric": "error",
            # "eta": 0.1,
            "seed": 42,
            "subsample": 1.0,
            "min_child_weight": 3,
            "max_depth": 8,
            "learning_rate": 0.05,
            "colsample_bytree": 0.6,
        }


class XGBRegressor(XGBoost):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = False
        self._params = {
            "objective": "reg:squarederror",
            # "eval_metric": "error",
            # "eta": 0.1,
            "seed": 42,
            "subsample": 1.0,
            "min_child_weight": 3,
            "max_depth": 8,
            "learning_rate": 0.05,
            "colsample_bytree": 0.6,
        }
