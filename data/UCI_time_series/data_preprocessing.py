###############################################################################
## File:         data_preprocessing.py
## Description:  Preprocess time series
## Authors:      Luong-Ha Nguyen & James-A. Goulet
## Created:      August 23, 2022
## Updated:      August 23, 2022
## Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
## Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import List
import pandas as pd


class TimeSeriesPrepocessing:
    """Preprocess the time series data to TAGI's format"""

    data_frame: pd.DataFrame

    def __init__(self, data_file: str) -> None:
        self.data_file = data_file

    @property
    def data_file(self) -> str:
        """Get data file"""
        return self._data_file

    @data_file.setter
    def data_file(self, value: str) -> None:
        """Set data file and raw data"""
        self._data_file = value
        self.data_frame = pd.read_csv(value)

    def get_time_series(self,
                        col_name: List[str],
                        time_col: str = "date_time") -> pd.DataFrame:
        """Get univariate time series"""

        sub_data_frame = self.data_frame[col_name]
        sub_data_frame = sub_data_frame.sort_values(time_col)
        sub_data_frame[time_col] = pd.to_datetime(sub_data_frame[time_col])
        sub_data_frame = sub_data_frame.iloc[
            sub_data_frame[time_col].drop_duplicates(keep='last').index]

        # Datetime range
        start_dt, end_dt = sub_data_frame.iloc[[0, -1]].values
        hourly_range = pd.date_range(start_dt[0], end_dt[0], freq="D")
        sub_data_frame = pd.DataFrame(hourly_range,
                                      columns=[time_col]).merge(sub_data_frame,
                                                                on=time_col,
                                                                how="outer")
        sub_data_frame = sub_data_frame.set_index(time_col)

        # Drop missing data
        sub_data_frame = sub_data_frame.dropna()

        return sub_data_frame

    def split_data(self, data_frame: pd.DataFrame, ratio: float = 0.9) -> None:
        """Split data into training and test sets"""

        # Split data
        last_idx = int(len(data_frame) * ratio)
        last_date_time = data_frame.index[last_idx]
        train_data = data_frame[(data_frame.index < last_date_time)]
        test_data = data_frame[(data_frame.index >= last_date_time)]

        # Save data as csv
        train_data.to_csv("train_data.csv", index=False)
        test_data.to_csv("test_data.csv", index=False)


def main():
    """API for preprocessing data"""
    data_file = "Metro_Interstate_Traffic_Volume.csv"
    col_name = ["date_time", "traffic_volume"]
    time_col = "date_time"
    ratio = 0.9

    dp = TimeSeriesPrepocessing(data_file=data_file)
    data_frame = dp.get_time_series(col_name=col_name, time_col=time_col)
    dp.split_data(data_frame=data_frame, ratio=ratio)


if __name__ == "__main__":
    main()
