from river.tree import HoeffdingTreeClassifier
from river.ensemble import AdaBoostClassifier as ExAdaBoostClassifier
from src.models.base_models.base_models import RiverStreamBaseModel


class AdaBoostClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExAdaBoostClassifier(
            model=(
                HoeffdingTreeClassifier(
                    split_criterion="gini",
                    grace_period=2000
                )
            ),
            n_models=5,
            seed=42
        )
