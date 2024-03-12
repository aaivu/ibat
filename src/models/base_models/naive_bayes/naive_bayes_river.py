from river.naive_bayes import (
    ComplementNB as ExComplementNB,
    BernoulliNB as ExBernoulliNB,
    MultinomialNB as ExMultinomialNB,
)
from src.models.base_models.base_models import RiverBatchBaseModel


class BernoulliNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExBernoulliNB()


class MultinomialNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExMultinomialNB()


class ComplementNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExComplementNB()
