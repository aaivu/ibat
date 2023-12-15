from abc import abstractmethod


class IStrategy:
    @abstractmethod
    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        pass
