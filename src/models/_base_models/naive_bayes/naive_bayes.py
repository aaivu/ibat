from sklearn.naive_bayes import (
    BernoulliNB as ExBernoulliNB,
    MultinomialNB as ExMultinomialNB,
)
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class BernoulliNB(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = ExBernoulliNB(**self._params)


class MultinomialNB(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "alpha": 1.0,
            "class_prior": None,
            "fit_prior": "None",
        }
        self._model = ExMultinomialNB(**self._params)
