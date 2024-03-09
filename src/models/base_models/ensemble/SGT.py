from river import tree
from src.models._base_models.river_base_model import RiverStreamBaseModel


class SGTRegressor(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = tree.SGTRegressor(
            delta=0.01,
            lambda_value=0.01,
            grace_period=20,
            feature_quantizer=tree.splitter.DynamicQuantizer(std_prop=0.1)
        )
