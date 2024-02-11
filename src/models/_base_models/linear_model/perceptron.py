from sklearn.linear_model import Perceptron
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class PerceptronClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = Perceptron(**self._params)


# Note: No regressor for Perceptron
