from river.tree import SGTRegressor as ExSGTRegressor
from river.tree.splitter import DynamicQuantizer
from src.models.base_models.base_models import RiverStreamBaseModel


class SGTRegressor(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = False
        self._params = {}
        self._model = ExSGTRegressor(
            delta=0.01,
            lambda_value=0.01,
            grace_period=20,
            feature_quantizer=DynamicQuantizer(std_prop=0.1),
        )
