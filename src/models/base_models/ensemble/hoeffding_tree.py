from river.preprocessing import StandardScaler
from river.tree import (
    HoeffdingAdaptiveTreeClassifier as ExHoeffdingAdaptiveTreeClassifier,
    HoeffdingTreeClassifier as ExHoeffdingTreeClassifier,
    HoeffdingTreeRegressor as ExHoeffdingTreeRegressor,
)
from src.models.base_models.base_models import RiverStreamBaseModel

"""
Tree-based models are popular due to their interpretability. Hoeffding Tree uses a tree data structure to model
the data. When a sample arrives, it traverses the tree until it reaches a leaf node. Internal nodes define the 
path for a data sample based on the values of its features. Leaf nodes are models that provide predictions for 
unlabeled-samples and can update their internal state using the labels from labeled samples.
"""


class HoeffdingTreeClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExHoeffdingTreeClassifier()


class HoeffdingTreeRegressor(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = False
        self._params = {}
        self._model = StandardScaler() | ExHoeffdingTreeRegressor(
            grace_period=100, model_selector_decay=0.9
        )


class HoeffdingAdaptiveTreeClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExHoeffdingAdaptiveTreeClassifier(seed=42)
