from pandas import DataFrame, Series
from river import stream
import pandas as pd

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


class RiverStreamBaseModel(IBaseModel):
    def __init__(self) -> None:
        self._model = None
        self._params = {}

    def fit(self, x, y):
        if self._model is not None:
            for xi, yi in stream.iter_pandas(X=x, y=y):
                self._model = self._model.learn_one(xi, yi)
            self._model.learn_many(x, y)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            for xi, yi in stream.iter_pandas(X=ni_x, y=ni_y):
                self._model = self._model.learn_one(xi, yi)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        results = pd.DataFrame()
        for index, row in x.iterrows():
            y = self._model.predict_one(row)
            temp_y = pd.DataFrame({'prediction': [y]})
            results = pd.concat([results, temp_y], axis=0)
        return results
