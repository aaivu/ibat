from pandas import DataFrame, Series
from src.models._base_models.ibase_model import IBaseModel


class RiverBatchBaseModel(IBaseModel):
    def __init__(self) -> None:
        self._model = None
        self._params = {}

    def fit(self, x, y):
        if self._model is not None:
            self._model.learn_many(x, y)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            self._model = self._model.learn_many(ni_x, ni_y)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        ser_prediction: Series = self._model.predict_many(x)
        return ser_prediction.to_frame(name="prediction")
