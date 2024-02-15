"""
SRP is an ensemble method that simulates bagging or random subspaces.
The default algorithm uses both bagging and random subspaces, namely Random Patches.
The default base estimator is a Hoeffding Tree, but other base estimators can be used (differently from random forest variations).
"""

from river.ensemble import SRPClassifier
from river.tree import HoeffdingTreeClassifier
from src.models._base_models.river_base_model import RiverStreamBaseModel


class StreamingRandomPatchesClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = SRPClassifier(model=HoeffdingTreeClassifier(),
                                    n_models=10,
                                    seed=42)
