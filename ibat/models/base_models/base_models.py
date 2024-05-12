from abc import ABC
from typing import Optional

from numpy import ndarray, unique
from pandas import concat, DataFrame, Series
from xgboost import Booster, DMatrix, train
from river.stream import iter_pandas
from ibat.models.base_models.ibase_model import IBaseModel


class BaseModel(IBaseModel, ABC):
    def __init__(self) -> None:
        self._is_classifier: Optional[bool] = None
        self._params = {}
        self._model = None

    @property
    def is_classifier(self) -> bool:
        return self._is_classifier

    @property
    def is_regressor(self) -> bool:
        return not self._is_classifier

    @property
    def model(self) -> any:
        if self._model is None:
            raise NotImplementedError()

        return self._model


class XGBoost(BaseModel, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[Booster] = None

    def fit(self, x, y):
        dtrain = DMatrix(data=x, label=y)
        self._model = train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=30,
        )

    def incremental_fit(self, ni_x, ni_y):
        dtrain = DMatrix(data=ni_x, label=ni_y)
        self._model = train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=self._model,
        )

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        data = DMatrix(data=x)
        np_prediction: ndarray = self._model.predict(data)
        return DataFrame({"prediction": np_prediction})


class SKLearnBaseModel(BaseModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, y):
        if self._model is not None:
            if self._is_classifier:
                self._model.partial_fit(x, y, classes=unique(y))
            else:
                self._model.partial_fit(x, y)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            self._model.partial_fit(ni_x, ni_y)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        np_prediction: ndarray = self._model.predict(x)
        return DataFrame({"prediction": np_prediction})


class RiverBatchBaseModel(BaseModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, y):
        if self._model is not None:
            self._model.learn_many(x, y)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            self._model.learn_many(ni_x, ni_y)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        ser_prediction: Series = self._model.predict_many(x)
        return ser_prediction.to_frame(name="prediction")


class RiverStreamBaseModel(BaseModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, y):
        if self._model is not None:
            for xi, yi in iter_pandas(x, y):
                self._model.learn_one(xi, yi)
        else:
            raise NotImplementedError("Model is Empty. Choose appropriate subclasses")

    def incremental_fit(self, ni_x, ni_y):
        if self._model is not None:
            for xi, yi in iter_pandas(ni_x, ni_y):
                self._model.learn_one(xi, yi)

    def predict(self, x) -> DataFrame:
        if len(x) == 0:
            return DataFrame(columns=["prediction"])

        pd_prediction = DataFrame()
        for index, row in x.iterrows():
            y = self._model.predict_one(row)
            pd_prediction = concat(
                [pd_prediction, DataFrame({"prediction": [y]})], axis=0
            )

        return pd_prediction
