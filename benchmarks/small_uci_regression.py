###############################################################################
# File:         small_uci_regression.py
# Description:  Running small uci datasets for user-defined epochs
# Authors:      Bhargob Deka & Luong-Ha Nguyen & James-A. Goulet
# Created:      April 08, 2024
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


from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp
from pytagi import Normalizer, Utils

## Load the data
data_names = ["Boston_housing"] # "Boston_housing","Concrete", "Energy", "Yacht", "Wine", "Kin8nm","Naval", "Power-plant","Protein"
# data_names = ["Boston_housing"]

for j in range(len(data_names)):

    # check if the results folder already exists; create it if does not exist or remove the existing one
    if not os.path.exists("benchmarks/logs/results_small_uci_regression/{}".format(data_names[j])):
        os.makedirs("benchmarks/logs/results_small_uci_regression/{}".format(data_names[j]))
    elif os.path.isfile("benchmarks/logs/results_small_uci_regression/{}/RMSEtest.txt".format(data_names[j])) and \
        os.path.isfile("benchmarks/logs/results_small_uci_regression/{}/LLtest.txt".format(data_names[j])) and \
        os.path.isfile("benchmarks/logs/results_small_uci_regression/{}/runtime_train.txt".format(data_names[j])):

        os.remove("benchmarks/logs/results_small_uci_regression/{}/RMSEtest.txt".format(data_names[j]))
        os.remove("benchmarks/logs/results_small_uci_regression/{}/LLtest.txt".format(data_names[j]))
        os.remove("benchmarks/logs/results_small_uci_regression/{}/runtime_train.txt".format(data_names[j]))


    # File paths for the results
    RESULTS_RMSEtest = "benchmarks/logs/results_small_uci_regression/"+data_names[j]+"/RMSEtest.txt"
    RESULTS_LLtest = "benchmarks/logs/results_small_uci_regression/"+data_names[j]+"/LLtest.txt"
    RESULTS_RUNTIME = "benchmarks/logs/results_small_uci_regression/"+data_names[j]+"/runtime_train.txt"

    # getting data name
    data_name = 'benchmarks/data/UCI/' + data_names[j]

    # load data
    data = np.loadtxt(data_name + '/data/data.txt')

    # We load the indexes for the features and for the target
    index_features = np.loadtxt(data_name +'/data/index_features.txt').astype(int)
    index_target   = np.loadtxt(data_name +'/data/index_target.txt').astype(int)

    # print(index_features)
    # print(index_target)

    # User-input (Default values)
    n_splits    = 20    # number of splits
    num_inputs  = len(index_features)     # 1 explanatory variable
    num_outputs = 1     # 1 predicted output
    num_epochs  = 100     # row for 40 epochs
    BATCH_SIZE  = 10     # batch size
    num_hidden_layers = 50

    # Change batch size for yacht
    if data_names[j] == "Yacht":
        BATCH_SIZE = 5


    # Change number of splits for Protein data to 5
    if data_names[j] == "Protein":
        n_splits = 5
        num_hidden_layers = 100

    EPOCHS = {}
    # Path to the directory containing optimal_epoch.txt files
    filepath_opt_epoch = "benchmarks/logs/results_small_uci_regression_earlystop/"+data_names[j]+"/optimal_epoch.txt"

    # Read the optimal epochs
    try:
        with open(filepath_opt_epoch, 'r') as file:
            epoch = int(file.read().strip())
            EPOCHS[data_names[j]] = epoch
    except FileNotFoundError:
        print(f"File not found: {filepath_opt_epoch}")

    # sigma V values for each dataset obtained via grid-search
    sigma_v_values = {"Boston_housing": 0.3, "Concrete": 0.3, "Energy": 0.1, "Yacht": 0.1, "Wine": 0.7, \
                        "Kin8nm": 0.3, "Naval": 0.6, "Power-plant": 0.2, "Protein": 0.7}

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
            self.layers         =  [1, 1, 1]
            self.nodes          =  [num_inputs, num_hidden_layers, 1]  # output layer = [mean, std]
            self.activations    =  [0, 4, 0]
            self.batch_size     =  BATCH_SIZE
            self.sigma_v        =  sigma_v_values[data_names[j]]
            self.sigma_v_min    =  0
            self.noise_gain     =  1.0
            # self.noise_type =   "homosce" # "heteros" or "homosce"
            self.init_method    =  "He"
            self.device         =  "cpu" # cpu
            self.early_stop     =  0 # early-stop not required for this as optimal epoch is used!!

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
    runtime_list = []

    # saving rmse and LL lists for each split
    rmse_splitlist = []
    LL_splitlist = []
    for i in range(n_splits):
        index_train = np.loadtxt(data_name +"/data/index_train_{}.txt".format(i)).astype(int)
        index_test = np.loadtxt(data_name +"/data/index_test_{}.txt".format(i)).astype(int)

        # print(index_train)
        # print(index_test)

        #Check for intersection of elements
        ind = np.intersect1d(index_train,index_test)
        if len(ind)!=0:
            print('Train and test indices are not unique')
            break

        # Train and Test data for the current split
        x_train = X[ index_train.tolist(), ]
        y_train = Y[ index_train.tolist() ]
        y_train = np.reshape(y_train,[len(y_train),1])
        x_test  = X[ index_test.tolist(), ]
        y_test  = Y[ index_test.tolist() ]
        y_test = np.reshape(y_test,[len(y_test),1])

        # print(x_train.shape)

        # Normalizer
        normalizer: Normalizer = Normalizer()

        x_mean, x_std = normalizer.compute_mean_std(x_train)
        y_mean, y_std = normalizer.compute_mean_std(y_train)




        x_train = normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        y_train = normalizer.standardize(data=y_train, mu=y_mean, std=y_std)
        x_test = normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = normalizer.standardize(data=y_test, mu=y_mean, std=y_std)

        # print(x_train.shape)
        # print(y_train.shape)



        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = create_data_loader(
            raw_input=x_test, raw_output=y_test, batch_size=BATCH_SIZE
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
        _, rmse_Epochlist, LL_Epochlist, _, _ = reg_task.train_UCI()

        # store rmse and LL lists for each split in rmse_splitlist and LL_splitlist
        rmse_splitlist.append(rmse_Epochlist)
        LL_splitlist.append(LL_Epochlist)

        # print(rmse_Epochlist)
        # print(LL_Epochlist)
        # time to run max epochs
        runtime = time.time()-start_time

        # Predict for one split
        mse, log_lik, rmse, _ = reg_task.predict_UCI()
        # Store the results
        mse_list.append(mse)
        log_lik_list.append(log_lik)
        rmse_list.append(rmse)
        runtime_list.append(runtime)


    mean_RMSE = np.mean(rmse_splitlist, axis=0)
    mean_LL   = np.mean(LL_splitlist,axis=0)
    # mean_RMSE = np.mean(rmsetests,axis=0)
    # print("best LL"+str(mean_ll[-1]))



    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(range(num_epochs), mean_RMSE)
    if os.path.exists(filepath_opt_epoch):
        ax[0].axvline(x=EPOCHS[data_names[j]], color='r', linestyle='--')  # Dashed line for optimal epoch on RMSE plot
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('RMSE')
    ax[1].scatter(range(num_epochs), mean_LL)
    if os.path.exists(filepath_opt_epoch):
        ax[1].axvline(x=EPOCHS[data_names[j]], color='r', linestyle='--')  # Dashed line for optimal epoch on RMSE plot
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Log-likelihood')
    # set the main title for the figure
    fig.suptitle(data_names[j])
    plt.savefig("benchmarks/logs/results_small_uci_regression/"+data_names[j]+"/RMSE_LL.png")

    # Print the average results
    print("Average MSE: ", np.mean(mse_list))
    print("Average Log-likelihood: ", np.mean(log_lik_list))
    print("Average RMSE: ", np.mean(rmse_list))
    print("Average Runtime: ", np.mean(runtime_list))

    # Save the average results
    with open(RESULTS_RMSEtest, "a") as file:
        file.write(str(np.mean(rmse_list)) + "\n")
    with open(RESULTS_LLtest, "a") as file:
        file.write(str(np.mean(log_lik_list)) + "\n")
    with open(RESULTS_RUNTIME, "a") as file:
        file.write(str(np.mean(runtime_list)) + "\n")




