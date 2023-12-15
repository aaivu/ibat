from typing import Optional

from numpy import ndarray
from pandas import DataFrame
from xgboost import Booster, DMatrix, train
from src.models._base_models.ibase_model import IBaseModel


class XGBoost(IBaseModel):
    def __init__(self) -> None:
        self._model: Optional[Booster] = None
        self._params = {}

    def fit(self, x, y):
        dtrain = DMatrix(data=x, label=y)
        self._model = train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=10,
        )

    def incremental_fit(self, ni_x, ni_y):
        dtrain = DMatrix(data=ni_x, label=ni_y)
        self._model = train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=5,
            xgb_model=self._model,
        )

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        data = DMatrix(data=x)
        np_prediction: ndarray = self._model.predict(data)
        return DataFrame({"prediction": np_prediction})


class XGBClassifier(XGBoost):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "objective": "binary:hinge",
            "eval_metric": "error",
            "eta": 0.1,
            "seed": 42,
        }


class XGBRegressor(XGBoost):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "objective": "reg:squarederror",
            "eval_metric": "error",
            "eta": 0.1,
            "seed": 42,
        }
