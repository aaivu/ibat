from enum import Enum
from typing import Optional

from numpy import ndarray
from pandas import DataFrame, read_csv
from torch import from_numpy, Tensor


class DatasetFileFormat(Enum):
    CSV = "csv"


class Dataset:
    def __init__(
        self,
        path: str,
        file_format: DatasetFileFormat = DatasetFileFormat.CSV
    ) -> None:
        self._path = path
        self._file_format = file_format
        self._dataframe = None

    def __getitem__(self, idx) -> Optional[DataFrame]:
        df = self.dataframe
        if df is not None:
            return df[idx]
        return None

    @property
    def dataframe(self) -> Optional[DataFrame]:
        if not self._dataframe:
            df = None
            if self._file_format == DatasetFileFormat.CSV:
                df = read_csv(self._path)
            self._dataframe = df

        return self._dataframe

    def as_numpy(self) -> Optional[ndarray]:
        df = self.dataframe
        if df is not None:
            return df.to_numpy()
        return None

    def as_tensor(self) -> Optional[Tensor]:
        df = self.dataframe
        if df is not None:
            # Assuming the DataFrame only has numerical data
            np_array = df.to_numpy()
            return from_numpy(np_array)
        return None


Bus654RunningTimes = Dataset(
    path="src/datasets/_datasets/bus_running_times_654.csv",
)

Bus654DwellTimes = Dataset(
    path="src/datasets/_datasets/bus_dwell_times_654.csv",
)
