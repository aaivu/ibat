from abc import abstractmethod
from typing import Any, Dict


class IStrategy:
    @abstractmethod
    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        pass

    @abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        pass
