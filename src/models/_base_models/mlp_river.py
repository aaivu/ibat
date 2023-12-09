from river import preprocessing as pp
from river import neural_net as nn
from river import optim

from src.models._base_models.river_learn_many_base_model import RiverBaseModelMany


class MLPRiverRegressor(RiverBaseModelMany):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = (
                pp.StandardScaler() |
                nn.MLPRegressor(
                    hidden_dims=(10,),
                    activations=(
                        nn.activations.ReLU,
                        nn.activations.ReLU,
                        nn.activations.ReLU
                    ),
                    optimizer=optim.SGD(1e-4),
                    seed=42
                )
        )
