from river import ensemble
from river import tree
from src.models._base_models.river_base_model import RiverStreamBaseModel


class AdaBoostClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = ensemble.AdaBoostClassifier(
            model=(
                tree.HoeffdingTreeClassifier(
                    split_criterion='gini',
                    grace_period=2000
                )
            ),
            n_models=5,
            seed=42
        )
