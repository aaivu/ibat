from sklearn.neural_network import (
    MLPClassifier as ExMLPClassifier,
    MLPRegressor as ExMLPRegressor,
)
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class MLPClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "hidden_layer_sizes": (100, 50),
            "random_state": "1",
        }
        self._model = ExMLPClassifier(**self._params)


class MLPRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "learning_rate": "adaptive",
        }
        self._model = ExMLPRegressor(**self._params)
