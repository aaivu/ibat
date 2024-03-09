from river.naive_bayes import BernoulliNB as RiverBernoulliNB, MultinomialNB as RiverMultinomialNB, ComplementNB as RiverComplementNB
from src.models._base_models.river_base_model import RiverBatchBaseModel


class BernoulliNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = RiverBernoulliNB()


class MultinomialNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = RiverMultinomialNB()


class ComplementNB(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = RiverComplementNB()