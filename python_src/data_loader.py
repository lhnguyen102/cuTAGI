###############################################################################
# File:         dataloader.py
# Description:  Prepare data for neural networks
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 13, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import numpy as np
import pandas as pd
import math


class RegressionDataLoader:
    """Load and format data that is feeded to the neural network.
     The user must provide the input and output data file in *csv"""

    def __init__(self, num_inputs: int, num_outputs: int,
                 batch_size: int) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_size = batch_size

    def process_data(self, x_train_file: str, y_train_file: str,
                     x_test_file: str, y_test_file: str) -> dict:
        """Process data from the csv file"""

        # Load data
        x_train = self.load_data_from_csv(x_train_file)
        y_train = self.load_data_from_csv(y_train_file)
        x_test = self.load_data_from_csv(x_test_file)
        y_test = self.load_data_from_csv(y_test_file)

        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = self.create_data_loader(raw_input=x_test,
                                                      raw_output=y_test)

        return data_loader

    def create_data_loader(self, raw_input: np.ndarray,
                           raw_output: np.ndarray) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = self.split_evenly(num_input_data, self.batch_size)

        # Remider indices
        rem_indices = self.split_reminder(num_input_data, self.batch_size)

        # Concat indices
        indices = np.concatenate((even_indices, rem_indices), axis=1)

        input_data = raw_input[indices]
        output_data = raw_output[indices]

        return list(zip(input_data, output_data))

    @staticmethod
    def split_data(data: int,
                   test_ratio: float = 0.2,
                   val_ratio: float = 0.0) -> dict:
        """Split data into training, validation, and test sets"""
        num_data = data.shape[1]
        splited_data = {}
        if val_ratio != 0.0:
            end_val_idx = num_data - int(test_ratio * num_data)
            end_train_idx = int(end_val_idx - val_ratio * end_val_idx)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = data[end_train_idx:end_val_idx]
            splited_data["test"] = data[end_val_idx:]
        else:
            end_train_idx = num_data - int(test_ratio * num_data)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = []
            splited_data["test"] = data[end_train_idx:]

        return splited_data

    @staticmethod
    def load_data_from_csv(data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, delimiter=",", skiprows=1)

        return data

    @staticmethod
    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(num_data)
        return np.split(indices, math.ceil(num_data / chunk_size))

    @staticmethod
    def split_reminder(num_data: int, chunk_size: int):
        """Pad the remider"""
        indices = np.arange(num_data)
        reminder_start = math.ceil(num_data / chunk_size)
        random_idx = np.random.choice(indices,
                                      size=num_data - reminder_start,
                                      replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx), axis=0)
