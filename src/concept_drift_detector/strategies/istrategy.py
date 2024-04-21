from abc import abstractmethod
from typing import Any, Dict

from pandas import DataFrame
# from src.models.base_models.base_models import BaseModel


class IStrategy:
    @abstractmethod
    def is_concept_drift_detected(
        self,
        model,
        ni_x: DataFrame,
        ni_y: DataFrame,
    ) -> bool:
        pass

    @abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        pass
