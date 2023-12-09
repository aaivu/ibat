from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from src.models._base_models.sklearn_base_model import SKLearnBaseModel


class IncBernoulliNB(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = BernoulliNB(**self._params)


class IncMultinomialNB(SKLearnBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {
            "alpha": 1.0,
            "class_prior": None,
            "fit_prior": "None"
        }
        self._model = MultinomialNB(**self._params)
