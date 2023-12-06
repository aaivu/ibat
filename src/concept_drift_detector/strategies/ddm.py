from src.concept_drift_detector.strategies  import IStrategy

class DDM(IStrategy):
    def is_concept_drift_detected(self, model, ni_x, ni_y):
        return True
