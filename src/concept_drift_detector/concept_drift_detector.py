from strategies.ddm import DDM

class CDD:
    def __init__(self):
        self._strategy = DDM()
       
    def is_concept_drift(self,model,ni_x,ni_y):
        return self.strategy.is_concept_drift(model,ni_x,ni_y)
    
    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy):
        self._strategy = new_strategy