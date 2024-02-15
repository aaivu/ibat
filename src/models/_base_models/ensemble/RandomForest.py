from river.forest import ARFClassifier
from src.models._base_models.river_base_model import RiverStreamBaseModel

"""
The 3 most important aspects of ARF are:

* inducing diversity through re-sampling
* inducing diversity through randomly selecting subsets of features for node splits
* drift detectors per base tree, which cause selective resets in response to drifts

It also allows training background trees, which start training if a warning is detected and replace the active tree if the warning escalates to a drift.
"""


class AdaptiveRandomForestClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = ARFClassifier(n_models=10)
