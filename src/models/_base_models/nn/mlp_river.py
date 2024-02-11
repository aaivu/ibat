from river.neural_net import MLPRegressor as ExMLPRegressor
from river.neural_net.activations import ReLU
from river.optim import SGD
from river.preprocessing import StandardScaler
from src.models._base_models.river_base_model import RiverBatchBaseModel


class MLPRegressor(RiverBatchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = StandardScaler() | ExMLPRegressor(
            hidden_dims=(10,),
            activations=(ReLU, ReLU, ReLU),
            optimizer=SGD(1e-4),
            seed=42,
        )
