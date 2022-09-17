###############################################################################
## File:         data_preprocessing.py
## Description:  Preprocess air passenger data
## Authors:      Luong-Ha Nguyen & James-A. Goulet
## Created:      September 17, 2022
## Updated:      September 17, 2022
## Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
## Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import pandas as pd


class DataPreprocessing:
    """Preprocess air passenger data"""
    data_frame: pd.DataFrame

    def __init__(self, data_file: str) -> None:
        self.data_file = data_file

    @property
    def data_file(self) -> str:
        """Get data file name"""
        return self._data_file

    @data_file.setter
    def data_file(self, value: str) -> None:
        """Set the data file"""
        self._data_file = value
        self.data_frame = pd.read_csv(value)

    def split_data(self, col_name: str, ratio: float = 0.8) -> None:
        """Split data into training and test sets"""

        # Split data
        last_idx = int(len(self.data_frame) * ratio)
        last_date_time = self.data_frame.index[last_idx]
        train_data = self.data_frame[col_name][(self.data_frame.index <
                                                last_date_time)]
        test_data = self.data_frame[col_name][(self.data_frame.index >=
                                               last_date_time)]

        # Save data as csv
        train_data.to_csv("train_air_passenger_data.csv", index=False)
        test_data.to_csv("test_air_passenger_data.csv", index=False)


def main():
    """API for preprocessing the air passenger data"""
    data_file = "AirPassengers.csv"
    col_name = "#Passengers"
    ratio = 0.8

    dp = DataPreprocessing(data_file=data_file)
    dp.split_data(col_name=col_name, ratio=ratio)


if __name__ == "__main__":
    main()
