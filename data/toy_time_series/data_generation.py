###############################################################################
## File:         data_generation.py
## Description:  Generate time series
## Authors:      Luong-Ha Nguyen & James-A. Goulet
## Created:      September 18, 2022
## Updated:      September 18, 2022
## Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
## License:      This code is released under the MIT License.
###############################################################################
from typing import Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataGeneration:
    """Generate the sin-like time series"""

    data_frame: pd.DataFrame

    def __init__(self, date_time: list) -> None:
        self.date_time = date_time

    @property
    def date_time(self) -> list:
        """Get datetime"""
        return self._date_time

    @date_time.setter
    def date_time(self, value: list) -> None:
        """Set datetime list"""
        self._date_time = value
        self.data_frame = self.create_dataframe(self._date_time)

    def create_dataframe(self, date_time: list) -> pd.DataFrame:
        """Create a time series x sin wave dataframe."""
        data_frame = pd.DataFrame(columns=["datetime", "sin"])
        data_frame["datetime"] = pd.date_range(
            start=date_time[0], end=date_time[1], freq="D"
        )
        data_frame["sin"] = 1 + np.sin(
            data_frame["datetime"].astype("int64")
            // 1e9
            * (2 * np.pi / (365.24 * 24 * 60 * 60))
        )
        data_frame["sin"] = data_frame["sin"] * 5
        data_frame = data_frame.set_index("datetime")

        return data_frame

    def split_data(self, col_name: Union[str, List[str]], ratio: float = 0.8) -> None:
        """Split data into training and test sets"""

        # Split data
        last_idx = int(len(self.data_frame) * ratio)
        last_date_time = self.data_frame.index[last_idx]
        train_idx = self.data_frame.index < last_date_time
        test_idx = self.data_frame.index >= last_date_time
        x_train_data = self.data_frame[col_name][train_idx]
        x_test_data = self.data_frame[col_name][test_idx]
        y_train_data = self.data_frame["sin"][train_idx]
        y_test_data = self.data_frame["sin"][test_idx]

        self.data_frame.reset_index(inplace=True)
        train_datetime = self.data_frame["datetime"][train_idx]
        test_datetime = self.data_frame["datetime"][test_idx]

        # Save data as csv
        x_train_data.to_csv("x_train_sin_data.csv", index=False)
        x_test_data.to_csv("x_test_sin_data.csv", index=False)
        y_train_data.to_csv("y_train_sin_data.csv", index=False)
        y_test_data.to_csv("y_test_sin_data.csv", index=False)
        train_datetime.to_csv("train_sin_datetime.csv", index=False)
        test_datetime.to_csv("test_sin_datetime.csv", index=False)

    def add_hourly_feature(self) -> None:
        """Add hourly features"""
        hour_sin = np.sin(2 * np.pi * (self.data_frame.index.hour.values / 24))
        hour_cos = np.cos(2 * np.pi * (self.data_frame.index.hour.values / 24))

        self.data_frame["hour_sin"] = hour_sin
        self.data_frame["hour_cos"] = hour_cos

    def add_weekly_feature(self) -> None:
        """Add weekly features"""

        week_sin = np.sin(2 * np.pi * (self.data_frame.index.isocalendar().day / 7))
        week_cos = np.cos(2 * np.pi * (self.data_frame.index.isocalendar().day / 7))
        self.data_frame["week_sin"] = week_sin
        self.data_frame["week_cos"] = week_cos

    def add_yearly_feature(self) -> None:
        """Add yearly feature"""

        year_sin = np.sin(2 * np.pi * (self.data_frame.index.isocalendar().day / 52))
        year_cos = np.cos(2 * np.pi * (self.data_frame.index.isocalendar().day / 52))
        self.data_frame["year_sin"] = year_sin
        self.data_frame["year_cos"] = year_cos

    def visualize_data(self) -> None:
        """Visualize time series"""
        _ = plt.figure(figsize=(12, 4))
        axes = plt.axes()
        axes.plot(self.data_frame["datetime"].values, self.data_frame["sin"].values)
        plt.show()


def main():
    """Generate time series"""
    date_time = ["2019-01-01", "2022-03-01"]
    # col_name = ["sin", "week_sin", "week_cos", "year_sin", "year_cos"]
    col_name = ["sin"]
    ratio = 0.8

    dg = DataGeneration(date_time)
    # dg.visualize_data()
    dg.add_weekly_feature()
    dg.add_yearly_feature()
    dg.split_data(col_name=col_name, ratio=ratio)


if __name__ == "__main__":
    main()
