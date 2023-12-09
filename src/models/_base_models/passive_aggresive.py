from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import PassiveAggressiveClassifier

from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class IncPassiveAggressiveClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": "42"
        }
        self._model = PassiveAggressiveClassifier(**self._params)


class IncPassiveAggressiveRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "C": 0.1,
            "max_iter": 1000,
            "random_state": "42"
        }
        self._model = PassiveAggressiveRegressor(**self._params)
