from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, HoeffdingTreeRegressor
from src.models._base_models.river_base_model import RiverStreamBaseModel
from river import preprocessing

"""
Tree-based models are popular due to their interpretability. Hoeffding Tree uses a tree data structure to model
 the data. When a sample arrives, it traverses the tree until it reaches a leaf node. Internal nodes define the 
 path for a data sample based on the values of its features. Leaf nodes are models that provide predictions for 
 unlabeled-samples and can update their internal state using the labels from labeled samples.

"""


class HoeffdingTreeClassifierRiver(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = HoeffdingTreeClassifier()


class HoeffdingTreeRegressorRiver(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = (
                preprocessing.StandardScaler() |
                HoeffdingTreeRegressor(
                    grace_period=100,
                    model_selector_decay=0.9
                )
        )


class HoeffdingAdaptiveTreeClassifierRiver(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = HoeffdingAdaptiveTreeClassifier(seed=42)
