###############################################################################
# File:         uci_prepocessing.py
# Description:  Preprocessing UCI datasets to data's format in TAGI
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      May 12, 2022
# Updated:      May 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
import pandas as pd
import numpy as np
import numpy.typing as npt

np.random.seed(1)


class UCIPreprocessing:
    """Preprocess the data that meet cuTAGI's requirements"""

    def __init__(
        self,
        data_folder: str,
        input_idx: list,
        output_idx: list,
        split_ratio: float = 0.9,
    ) -> None:
        self.data_folder = data_folder
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.split_ratio = split_ratio

        self.raw_data: npt.NDArray | None = None
        self.train_input_data: npt.NDArray | None = None
        self.train_output_data: npt.NDArray | None = None
        self.test_input_data: npt.NDArray | None = None
        self.test_output_data: npt.NDArray | None = None

    def load_raw_data(self) -> npt.NDArray:
        """Load the raw data"""

        file_name_path = f"./data/UCI/{self.data_folder}/data/data.txt"
        self.raw_data = np.loadtxt(file_name_path)

    def split_data(self) -> None:
        """Split raw data into the train and test data"""

        # Number of data
        n, col = self.raw_data.shape

        # Get train and test indices
        perm_idx = np.random.choice(range(n), n, replace=False)
        # perm_idx = np.arange(n)
        end_train_idx = round(n * self.split_ratio)
        train_idx = perm_idx[0:end_train_idx]
        test_idx = perm_idx[end_train_idx:n]

        # Train data
        train_raw_data = self.raw_data[train_idx, :]
        self.train_input_data = train_raw_data[:, np.array(self.input_idx)]
        self.train_output_data = train_raw_data[:, np.array(self.output_idx)]

        # Test data
        test_raw_data = self.raw_data[test_idx, :]
        self.test_input_data = test_raw_data[:, np.array(self.input_idx)]
        self.test_output_data = test_raw_data[:, np.array(self.output_idx)]

    def to_csv(self) -> None:
        """Save as csv file"""

        # Input data
        train_input_dict = {}
        test_input_dict = {}
        input_data_label = []
        for i in range(len(self.input_idx)):
            col_label = f"x_{i + 1}"
            train_input_dict[col_label] = self.train_input_data[:, i]
            test_input_dict[col_label] = self.test_input_data[:, i]
            input_data_label.append(col_label)

        # Save input data as csv file
        train_input_path = f"./data/UCI/{self.data_folder}/x_train.csv"
        train_input_df = pd.DataFrame(train_input_dict, columns=input_data_label)
        train_input_df.to_csv(train_input_path, index=False, header=True)

        test_input_path = f"./data/UCI/{self.data_folder}/x_test.csv"
        test_input_df = pd.DataFrame(test_input_dict, columns=input_data_label)
        test_input_df.to_csv(test_input_path, index=False, header=True)

        # Output data
        train_output_dict = {}
        test_output_dict = {}
        output_data_label = []
        for i in range(len(self.output_idx)):
            col_label = f"y_{i + 1}"
            train_output_dict[col_label] = self.train_output_data[:, i]
            test_output_dict[col_label] = self.test_output_data[:, i]
            output_data_label.append(col_label)

        # Save output data as csv file
        train_output_path = f"./data/UCI/{self.data_folder}/y_train.csv"
        train_output_df = pd.DataFrame(train_output_dict, columns=output_data_label)
        train_output_df.to_csv(train_output_path, index=False, header=True)

        test_output_path = f"./data/UCI/{self.data_folder}/y_test.csv"
        test_output_df = pd.DataFrame(test_output_dict, columns=output_data_label)
        test_output_df.to_csv(test_output_path, index=False, header=True)


def main():
    # User input
    input_idx = list(np.arange(13))
    output_idx = [13]
    split_ratio = 0.9
    prep = UCIPreprocessing(
        data_folder="Boston_housing",
        input_idx=input_idx,
        output_idx=output_idx,
        split_ratio=split_ratio,
    )
    prep.load_raw_data()
    prep.split_data()
    prep.to_csv()


if __name__ == "__main__":
    main()
