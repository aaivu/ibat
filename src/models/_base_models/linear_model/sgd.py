from sklearn.linear_model import (
    SGDClassifier as ExSGDClassifier,
    SGDRegressor as ExSGDRegressor,
)
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class SGDClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "loss": "log_loss",
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExSGDClassifier(**self._params)


class SGDRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExSGDRegressor(**self._params)
