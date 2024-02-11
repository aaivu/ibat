from sklearn.linear_model import (
    PassiveAggressiveClassifier as ExPassiveAggressiveClassifier,
    PassiveAggressiveRegressor as ExPassiveAggressiveRegressor,
)
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class PassiveAggressiveClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExPassiveAggressiveClassifier(**self._params)


class PassiveAggressiveRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "C": 0.1,
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExPassiveAggressiveRegressor(**self._params)
