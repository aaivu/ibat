from src.concept_drift_detector.strategies.istrategy import IStrategy


class CDD:
    def __init__(self, strategy: IStrategy):
        self._strategy = strategy

    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        return self.strategy.is_concept_drift_detected(model, ni_x, ni_y)

    @property
    def strategy(self) -> IStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: IStrategy):
        self._strategy = new_strategy
