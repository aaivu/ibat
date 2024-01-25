from numpy import ndarray
from pandas import DataFrame
from src.models._base_models.ibase_model import IBaseModel


class SKLearnBaseModel(IBaseModel):
    def __init__(self) -> None:
        self._model = None
        self._params = {}

    def fit(self, x, y):
        if self._model is not None:
            self._model.partial_fit(x, y)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            self._model = self._model.partial_fit(ni_x, ni_y)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])
        np_prediction: ndarray = self._model.predict(x)
        return DataFrame({"prediction": np_prediction})
