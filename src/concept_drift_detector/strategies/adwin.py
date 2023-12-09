from river import drift
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Adwin():

    def __init__(self,delta):
        self.delta = delta

    def is_concept_drift_detected(self, model, ni_x, ni_y):
        stream = ni_x
        drift_detector = drift.ADWIN(delta=self.delta)
        drifts = []
        for i, val in enumerate(stream):
            drift_detector.update(val) 
            if drift_detector.change_detected:
                is_detected = drift_detector.change_detected
                print(f'Change detected at index {i}')
                drifts.append(i)
        plot_data(stream, drifts)
        
        return is_detected

def plot_data(stream, drifts=None):
    fig = plt.figure(figsize=(7,3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    
    plt.grid()
    plt.plot(stream, label='Stream')
    if drifts is not None:
        for drift_detected in drifts:
            plt.axvline(drift_detected, color='red')
    plt.xlabel(f'date')
    plt.ylabel(f'mean arrival time')
    plt.show()
