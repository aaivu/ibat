from abc import abstractmethod


class IBaseModel:
    @abstractmethod
    def fit(self, x, y) -> None:
        pass

    @abstractmethod
    def incremental_fit(self, ni_x, ni_y) -> None:
        pass

    @abstractmethod
    def predict(self, x) -> None:
        pass
