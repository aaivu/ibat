from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron

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


class IncSGDClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "loss": 'log_loss',
            "max_iter": 1000,
            "random_state": "42"
        }
        self._model = SGDClassifier(**self._params)


class IncSGDRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "max_iter": 1000,
            "random_state": "42"
        }
        self._model = SGDRegressor(**self._params)


class IncPerceptronClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "max_iter": 1000,
            "random_state": "42"
        }
        self._model = Perceptron(**self._params)

# No regressor for Perceptron
