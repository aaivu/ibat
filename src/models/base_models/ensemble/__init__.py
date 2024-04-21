from .adaboost import AdaBoostClassifier
from .decision_tree import DecisionTreeClassifier
from .hoeffding_tree import (
    HoeffdingAdaptiveTreeClassifier,
    HoeffdingTreeClassifier,
    HoeffdingTreeRegressor,
)

# from .random_forest import ARFClassifier, ARFRegressor
from .sgt import SGTRegressor
from .streaming_random_patches import SRPClassifier
from .xgboost import XGBClassifier, XGBRegressor
from . import (
    adaboost,
    decision_tree,
    hoeffding_tree,
    random_forest,
    sgt,
    streaming_random_patches,
    xgboost,
)
