from src.concept_drift_detector.strategies  import IStrategy
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from frouros.detectors.concept_drift import DDM as fDDM, DDMConfig
from frouros.metrics import PrequentialError

class DDM(IStrategy):

    def __init__(self,warning_level, drift_level, min_num_instances):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_num_instances = min_num_instances
        
        self.config = DDMConfig( 
            warning_level=self.warning_level, 
            drift_level=self.drift_level,
            min_num_instances=self.min_num_instances
        )

        self.detector = fDDM(config=self.config)
        self.metric = PrequentialError(alpha=1.0) 

    def is_concept_drift_detected(self, model, ni_x, ni_y):
        drift_flag = False
        idx_drift, idx_warning = [], []
        for (i, X), y in zip( ni_x.iterrows(), ni_y ):

            data = {
                'year': [X['year']],
                'month':  [X['month']],
                'day': [X['day']]
            }
            X = pd.DataFrame.from_dict(data)
            y_pred = model.predict(X)
            error = mean_absolute_percentage_error([y] , y_pred)
            metric_error = self.metric(error_value=error)
            _ = self.detector.update(value=error)
            status = self.detector.status            
            
            if status["drift"] and not drift_flag:
                drift_flag = True
                idx_drift.append(i)
                print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
            if status["warning"]:
                print(f'warning detected: {i} error={error:.4f} Accuracy : {1 - metric_error:.4f} ')
                idx_warning.append(i)
        if not drift_flag:
            print("No concept drift detected")
        
        print(f"Final accuracy: {1 - metric_error:.4f}\n")
        print ('warning index : ',idx_warning)
        print ('drift index : ',idx_drift)
        return idx_drift,idx_warning
    

