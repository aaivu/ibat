from strategies.ddm import DDM


class CDD:
    def __init__(self, model, X_test,y_test):
        self._strategy = DDM()
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def is_concept_drift(self,model,X_test,y_test):
        return self.strategy.is_concept_drift(model,X_test,y_test)
    

    # Getter method for _strategy
    @property
    def strategy(self):
        return self._strategy

    # Setter method for _strategy
    @strategy.setter
    def strategy(self, new_strategy):
        self._strategy = new_strategy