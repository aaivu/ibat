from sklearn.linear_model import Perceptron
from ibat.models.base_models.base_models import SKLearnBaseModel


class PerceptronClassifier(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        self._model = Perceptron(**self._params)


# Note: No regressor for Perceptron
