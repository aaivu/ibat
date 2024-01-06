from typing import Any, Dict

from river.drift import ADWIN as rADWIN
from matplotlib import gridspec, pyplot as plt
from src.concept_drift_detector.strategies.istrategy import IStrategy


class ADWIN(IStrategy):
    def __init__(self, delta: float) -> None:
        self._delta = delta
        self._adwin = rADWIN(delta=delta)

    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        is_detected = False
        drifts = []

        for i, x in enumerate(ni_x):
            self._adwin.update(x)
            if self._adwin.drift_detected:
                is_detected = self._adwin.drift_detected
                print(f"Change detected at index {i}, input value: {x}")
                drifts.append(i)

        plot_data(ni_x, drifts)

        return is_detected

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "Delta": self._delta,
        }


def plot_data(stream, drifts=None):
    fig = plt.figure(figsize=(7, 3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    plt.grid()
    plt.plot(stream, label="Stream")
    if drifts is not None:
        for drift_detected in drifts:
            plt.axvline(drift_detected, color="red")
    plt.xlabel(f"date")
    plt.ylabel(f"mean arrival time")
    plt.savefig("adwin.jpg", dpi=200)
