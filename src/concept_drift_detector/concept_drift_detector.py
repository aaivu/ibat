class CDD:
    def __init__(self,strategy ):
        self._strategy = strategy
       
    def is_concept_drift_detected(self, model, ni_x, ni_y):
        return self.strategy.is_concept_drift_detected(model, ni_x, ni_y)
    
    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy):
        self._strategy = new_strategy