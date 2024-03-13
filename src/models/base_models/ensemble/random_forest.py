# from river.forest import (ARFClassifier as ExARFClassifier, ARFRegressor as ExARFRegressor)
# from river.preprocessing import StandardScaler
# from src.models.base_models.base_models import RiverStreamBaseModel
#
# """
# The 3 most important aspects of ARF are:
#
# * inducing diversity through re-sampling
# * inducing diversity through randomly selecting subsets of features for node splits
# * drift detectors per base tree, which cause selective resets in response to drifts
#
# It also allows training background trees, which start training if a warning is detected and replace the active tree if the warning escalates to a drift.
# """
#
#
# class ARFClassifier(RiverStreamBaseModel):
#     def __init__(self) -> None:
#         super().__init__()
#         self._is_classifier = True
#         self._params = {}
#         self._model = ExARFClassifier(n_models=10)
#
#
# class ARFRegressor(RiverStreamBaseModel):
#     def __init__(self) -> None:
#         super().__init__()
#         self._is_classifier = False
#         self._params = {}
#         self._model = (
#                 StandardScaler() |
#                 ExARFRegressor(seed=42)
#         )
