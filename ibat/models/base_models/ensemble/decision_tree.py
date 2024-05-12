from river.tree import ExtremelyFastDecisionTreeClassifier
from ibat.models.base_models.base_models import RiverStreamBaseModel


class DecisionTreeClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_classifier = True
        self._params = {}
        self._model = ExtremelyFastDecisionTreeClassifier(
            grace_period=100,
            nominal_attributes=["elevel", "car", "zipcode"],
            min_samples_reevaluate=100,
        )
