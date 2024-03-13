from river.ensemble import SRPClassifier as ExSRPClassifier
from river.tree import HoeffdingTreeClassifier
from src.models.base_models.base_models import RiverStreamBaseModel

"""
SRP is an ensemble method that simulates bagging or random subspaces.
The default algorithm uses both bagging and random subspaces, namely Random Patches.
The default base estimator is a Hoeffding Tree, but other base estimators can be used (differently from random forest variations).
"""


class SRPClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExSRPClassifier(
            model=HoeffdingTreeClassifier(),
            n_models=10,
            seed=42,
        )
