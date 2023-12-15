from river.drift import PageHinkley as rPageHinkley
from matplotlib import gridspec, pyplot as plt
from src.concept_drift_detector.strategies.istrategy import IStrategy


class PageHinkley(IStrategy):
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._ph = rPageHinkley(threshold=threshold)

    def is_concept_drift_detected(self, model, ni_x, ni_y) -> bool:
        is_detected = False
        drifts = []

        for i, x in enumerate(ni_x):
            self._ph.update(x)
            if self._ph.drift_detected:
                is_detected = self._ph.drift_detected
                print(f"Change detected at index {i}, input value: {x}")
                drifts.append(i)

        plot_data(ni_x, drifts)

        return is_detected


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
    plt.savefig("ph.jpg", dpi=200)
