from src.models._base_models.river_learn_many_base_model import RiverBaseModelMany
from river import linear_model
from river import preprocessing
from river import optim


class LinearRegression(RiverBaseModelMany):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "intercept_lr": 0.1
        }
        self._model = (
                preprocessing.StandardScaler() |
                linear_model.LinearRegression(**self._params)
        )


class LogisticRegressionClassifier(RiverBaseModelMany):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "optimizer": optim.SGD(.1)
        }
        self._model = (
                preprocessing.StandardScaler() |
                linear_model.LogisticRegression(**self._params)
        )
