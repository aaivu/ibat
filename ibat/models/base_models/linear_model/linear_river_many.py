from river.linear_model import (
    LinearRegression as ExLinearRegression,
    LogisticRegression as ExLogisticRegression,
)
from river.optim import SGD
from river.preprocessing import StandardScaler
from ibat.models.base_models.base_models import RiverBatchBaseModel


class LinearRegression(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = False
        self._params = {"intercept_lr": 0.1}
        self._model = StandardScaler() | ExLinearRegression(**self._params)


class LogisticRegression(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {"optimizer": SGD(0.1)}
        self._model = StandardScaler() | ExLogisticRegression(**self._params)
