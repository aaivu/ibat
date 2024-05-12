from sklearn.linear_model import (
    SGDClassifier as ExSGDClassifier,
    SGDRegressor as ExSGDRegressor,
)
from ibat.models.base_models.base_models import SKLearnBaseModel


class SGDClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {
            "loss": "hinge",
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExSGDClassifier(**self._params)


class SGDRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = False
        self._params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = ExSGDRegressor(**self._params)
