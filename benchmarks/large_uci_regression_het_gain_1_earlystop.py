###############################################################################
# File:         large_uci_regression_het_gain_1_earlystop.py
# Description:  Running large uci datasets with heteroscedasticity and early-stopping
# Authors:      Bhargob Deka & Luong-Ha Nguyen & James-A. Goulet
# Created:      April 13, 2024
# Updated:      --
# Contact:      bhargobdeka11@gmail.com & luongha.nguyen@gmail.com
#               & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################


## Libraries
import time
import csv
import os
import pandas as pd
import numpy as np
from typing import Union, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp

## Load the data
data_names = ["elevators", "skillcraft", "pol", "keggdirected", "keggundirected"] #"elevators","skillcraft","pol", "keggdirected", "keggundirected"

# setting early stopping
early_stop = 1

for j in range(len(data_names)):

    # check if the results folder already exists; create it if does not exist or remove the existing one
    if not os.path.exists("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}".format(data_names[j])):
        os.makedirs("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}".format(data_names[j]))
    elif os.path.isfile("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/RMSEtest.txt".format(data_names[j])) and \
        os.path.isfile("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/LLtest.txt".format(data_names[j])) and \
        os.path.isfile("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/optimal_epoch.txt".format(data_names[j])):

        os.remove("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/RMSEtest.txt".format(data_names[j]))
        os.remove("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/LLtest.txt".format(data_names[j]))
        os.remove("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/runtime_train.txt".format(data_names[j]))
        os.remove("benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/{}/optimal_epoch.txt".format(data_names[j]))

    # File paths for the results
    RESULTS_RMSEtest = "benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/"+data_names[j]+"/RMSEtest.txt"
    RESULTS_LLtest = "benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/"+data_names[j]+"/LLtest.txt"
    RESULTS_RUNTIME = "benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/"+data_names[j]+"/runtime_train.txt"
    RESULTS_EPOCH = "benchmarks/logs/results_large_uci_regression_het_gain_1_earlystop/"+data_names[j]+"/optimal_epoch.txt"

    # getting data name
    data_name = 'benchmarks/data/UCI/' + data_names[j]

    # load data
    data = np.loadtxt(data_name + '/data.txt')

    # We load the indexes for the features and for the target
    index_features = np.loadtxt(data_name +'/index_features.txt').astype(int)
    index_target   = np.loadtxt(data_name +'/index_target.txt').astype(int)

    # print(index_features)
    # print(index_target)

    # User-input (Default values)
    n_splits    = 10    # number of splits
    num_inputs  = len(index_features)     # 1 explanatory variable
    num_outputs = 1      # 1 predicted output
    num_epochs  = 100    # row for 40 epochs
    BATCH_SIZE  = 10     # batch size
    num_hidden_layers = 50

    # Gain values for each dataset as provided in Table H.1 of the Supplementary material
    # of the paper "Analytically tractable heteroscedastic uncertainty quantification
    # in Bayesian neural networks for regression tasks"
    # OUT_GAIN = {"elevators": 1, "pol": 0.5, "skillcraft": 0.1, "keggdirected": 0.1,\
    #     "keggundirected": 0.1}
    # NOISE_GAIN = {"elevators": 0.1, "pol": 0.001, "skillcraft": 0.1, "keggdirected": 0.0001,\
    #     "keggundirected": 0.1}

    def remove_outliers(data, threshold=5):
        """
        Removes outliers from the dataset using Z-score.

        Parameters:
            data (numpy.ndarray): The dataset.
            threshold (float): The threshold for the Z-score. Data points with a Z-score higher than this threshold will be considered outliers. Default is 3.

        Returns:
            numpy.ndarray: Cleaned dataset with outliers removed.
        """
        z_scores = np.abs(stats.zscore(data))
        filtered_data = data[(z_scores < threshold).all(axis=1)]
        return filtered_data

    # remove outliers if required
    if data_names[j]=="skillcraft":
        data = remove_outliers(data, threshold=5)
        num_epochs = 20
    if data_names[j]=="pol":
        num_epochs = 200

    # Input data and output data
    X = data[ : , index_features.tolist() ]
    Y = data[ : , index_target.tolist() ]
    input_dim = X.shape[1]


    ## classes
    class HeterosUCIMLP(NetProp):
        """Multi-layer preceptron for regression task where the
        output's noise varies overtime"""

        def __init__(self) -> None:
            super().__init__()
            self.layers         =  [1, 1, 1, 1, 1, 1]
            self.activations    =  [0, 4, 4, 4, 4, 0]
            if data_names[j]=="skillcraft":
                self.nodes          =  [num_inputs, 1000, 500, 50, 2]  # output layer = [mean, std]
                self.layers         =  [1, 1, 1, 1, 1]
                self.activations    =  [0, 4, 4, 4, 0]
            else:
                self.nodes          =  [num_inputs, 1000, 1000, 500, 50, 2]  # output layer = [mean, std]

            self.batch_size     =  BATCH_SIZE
            self.sigma_v        =  0 # sigma_v_values[data_names[j]]
            self.sigma_v_min    =  0
            self.out_gain       =  1 #OUT_GAIN[data_names[j]]
            self.noise_gain     =  1 #NOISE_GAIN[data_names[j]]
            self.noise_type     =   "heteros" # "heteros" or "homosce"
            self.init_method    =  "He"
            self.device         =  "cuda" # cpu, cuda
            self.early_stop     =  early_stop
            self.delta          =  0.01
            self.patience       =  5


    ## Functions
    def create_data_loader(raw_input: np.ndarray, raw_output: np.ndarray, batch_size) -> list:
            """Create dataloader based on batch size"""
            num_input_data = raw_input.shape[0]
            num_output_data = raw_output.shape[0]
            assert num_input_data == num_output_data

            # Even indices
            even_indices = split_evenly(num_input_data, batch_size)

            if np.mod(num_input_data, batch_size) != 0:
                # Remider indices
                rem_indices = split_reminder(num_input_data, batch_size)
                even_indices.append(rem_indices)

            indices = np.stack(even_indices)
            input_data = raw_input[indices]
            output_data = raw_output[indices]
            dataset = []
            for x_batch, y_batch in zip(input_data, output_data):
                dataset.append((x_batch, y_batch))
            return dataset


    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))

        return np.split(indices, int(np.floor(num_data / chunk_size)))

    def split_reminder(num_data: int, chunk_size: int):
            """Pad the reminder"""
            indices = np.arange(num_data)
            reminder_start = int(num_data - np.mod(num_data, chunk_size))
            num_samples = chunk_size - (num_data - reminder_start)
            random_idx = np.random.choice(indices, size=num_samples, replace=False)
            reminder_idx = indices[reminder_start:]

            return np.concatenate((random_idx, reminder_idx))


    mse_list = []
    log_lik_list = []
    rmse_list = []
    normal_log_lik_list = []
    runtime_list = []

    # saving rmse and LL lists for each split
    rmse_splitlist = []
    LL_splitlist = []
    normal_LL_splitlist = []
    stopEpoch_list = []
    for i in range(n_splits):

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
        y_train = np.reshape(y_train,[len(y_train),1])
        y_test = np.reshape(y_test,[len(y_test),1])


        # splitting training data into train and val data if early-stopping is needed
        if early_stop == 1:
           x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        else:
            x_val, y_val = [], []

        # print(x_train.shape)

        # Normalizer
        from pytagi import Normalizer, Utils

        normalizer: Normalizer = Normalizer()

        x_mean, x_std = normalizer.compute_mean_std(x_train)
        y_mean, y_std = normalizer.compute_mean_std(y_train)

        # change x_std to 1 when the stdv. values are zero or very small, i.e., < 1e-05
        idx = x_std == 0
        x_std[x_std < 1e-05] = 1
        x_std[idx] = 1

        # normalizing data
        x_train = normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        y_train = normalizer.standardize(data=y_train, mu=y_mean, std=y_std)
        x_test = normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = normalizer.standardize(data=y_test, mu=y_mean, std=y_std)


        # normalize val data
        if early_stop==1:
            x_val = normalizer.standardize(data=x_val, mu=x_mean, std=x_std)
            y_val = normalizer.standardize(data=y_val, mu=y_mean, std=y_std)



        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = create_data_loader(
            raw_input=x_test, raw_output=y_test, batch_size=BATCH_SIZE
        )
        if early_stop==1:
            data_loader["val"] = create_data_loader(
            raw_input=x_val, raw_output=y_val, batch_size=BATCH_SIZE
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = y_mean
        data_loader["y_norm_param_2"] = y_std

        # print(data_loader["train"][0].shape)


        # Model
        net_prop = HeterosUCIMLP()


        reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                        num_outputs=num_outputs,
                                        batch_size=net_prop.batch_size)


        reg_task = Regression(num_epochs=num_epochs,
                        data_loader=data_loader,
                        net_prop=net_prop)


        # Train the network
        start_time = time.time()
        _, _, _, _, stop_epoch = reg_task.train_UCI()

        # storing stop_epoch for each split
        stopEpoch_list.append(stop_epoch)

        # Time to run max epochs
        runtime = time.time()-start_time


        # Predict for one split
        mse, log_lik, rmse, normal_LL = reg_task.predict_UCI()
        # Store the results
        mse_list.append(mse)
        log_lik_list.append(log_lik)
        rmse_list.append(rmse)
        normal_log_lik_list.append(normal_LL)
        runtime_list.append(runtime)

    # Print the average results
    print("Average MSE: ", np.mean(mse_list))
    print("Average Normalised Log-likelihood: ", np.mean(normal_log_lik_list))
    print("Average RMSE: ", np.mean(rmse_list))
    print("Average Runtime: ", np.mean(runtime_list))
    print("Average optimal epoch: ", np.mean(stopEpoch_list))

    # Save the average results
    with open(RESULTS_RMSEtest, "a") as file:
        file.write(str(np.mean(rmse_list)) + "\n")
    with open(RESULTS_LLtest, "a") as file:
        file.write(str(np.mean(normal_log_lik_list)) + "\n")
    with open(RESULTS_RUNTIME, "a") as file:
        file.write(str(np.mean(runtime_list)) + "\n")
    with open(RESULTS_EPOCH, "a") as file:
        file.write(str(round(np.mean(stopEpoch_list))) + "\n")