from river import tree
from src.models._base_models.river_base_model import RiverStreamBaseModel


class ExtremelyFastDecisionTreeClassifier(RiverStreamBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._params = {}
        self._model = tree.ExtremelyFastDecisionTreeClassifier(
            grace_period=100,
            nominal_attributes=['elevel', 'car', 'zipcode'],
            min_samples_reevaluate=100
        )
