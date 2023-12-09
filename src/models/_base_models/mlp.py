from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class IncMLPClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "hidden_layer_sizes": (100, 50),
            "random_state": "1"
        }
        self._model = MLPClassifier(**self._params)


class IncMLPRegressor(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "hidden_layer_sizes": (100, 50),
            "activation": 'relu',
            "learning_rate": 'adaptive'
        }
        self._model = MLPRegressor(**self._params)
