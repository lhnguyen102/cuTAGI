## Libraries
import time
import csv
import os
import pandas as pd
import numpy as np
from typing import Union, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
# go one level down to import the data_loader and regression classes
# os.chdir('..')
import sys
print(sys.path)
sys.path.append('/home/bd/projects/cuTAGI') # always append the path to the root directory

from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp

## Load the data
# data_names = ["Wine", \
#               "Kin8nm","Naval",\
#               "Power-plant","Protein"]
data_names = ["Boston_housing"] # "Concrete", "Energy", "Yacht", "Wine", "Kin8nm","Naval", "Power-plant","Protein"

for j in range(len(data_names)):

    # check if the results folder already exists; create it if does not exist or remove the existing one
    if not os.path.exists("results_small_UCI_TAGI_AGVI_Het/{}".format(data_names[j])):
        os.makedirs("results_small_UCI_TAGI_AGVI_Het/{}".format(data_names[j]))
    elif os.path.isfile("results_small_UCI_TAGI_AGVI_Het/{}/RMSEtest.txt".format(data_names[j])) and \
        os.path.isfile("results_small_UCI_TAGI_AGVI_Het/{}/LLtest.txt".format(data_names[j])) and \
        os.path.isfile("results_small_UCI_TAGI_AGVI_Het/{}/runtime_train.txt".format(data_names[j])):

        os.remove("results_small_UCI_TAGI_AGVI_Het/{}/RMSEtest.txt".format(data_names[j]))
        os.remove("results_small_UCI_TAGI_AGVI_Het/{}/LLtest.txt".format(data_names[j]))
        os.remove("results_small_UCI_TAGI_AGVI_Het/{}/runtime_train.txt".format(data_names[j]))


    # File paths for the results
    RESULTS_RMSEtest = "results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/RMSEtest.txt"
    RESULTS_LLtest = "results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/LLtest.txt"
    RESULTS_RUNTIME = "results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/runtime_train.txt"
    RESULTS_LL_learning_curve = "results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/learning_curve_LL.txt"
    RESULTS_RMSE_learning_curve = "results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/learning_curve_RMSE.txt"

    # getting data name
    data_name = 'data/UCI/' + data_names[j]

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
    BATCH_SIZE  = 32     # batch size
    num_hidden_layers = 50

    # Gain values for each dataset
    # OUT_GAIN = {"Boston_housing": 0.5, "Concrete": 0.5, "Energy": 0.5, "Yacht": 1, "Wine": 0.1, \
    #                     "Kin8nm": 0.5, "Naval": 0.5, "Power-plant": 0.5, "Protein": 0.5}
    NOISE_GAIN = {"Boston_housing": 1, "Concrete": 0.05, "Energy": 0.1, "Yacht": 0.1, "Wine": 0.01, \
                        "Kin8nm": 0.1, "Naval": 0.01, "Power-plant": 0.001, "Protein": 0.1}

    # Change batch size for wine and yacht
    if data_names[j] == "Yacht":
        BATCH_SIZE = 5
    if data_names[j] == "Energy":
        BATCH_SIZE = 10

    # Change number of splits for Protein data to 5
    if data_names[j] == "Protein":
        n_splits = 5
        num_hidden_layers = 100


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
            self.nodes          =  [num_inputs, num_hidden_layers, 2]  # output layer = [mean, std]
            self.activations    =  [0, 4, 0]
            self.batch_size     =  BATCH_SIZE
            self.sigma_v        =  0 # sigma_v_values[data_names[j]]
            self.sigma_v_min    =  0
            # self.out_gain       = OUT_GAIN[data_names[j]]
            self.noise_gain     =  NOISE_GAIN[data_names[j]]
            self.noise_type     =   "heteros" # "heteros" or "homosce"
            self.init_method    =  "He"
            self.device         =  "cpu" # cpu

    ## Functions$
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
        y_train = np.reshape(y_train,[len(y_train),1]) #BD
        x_test  = X[ index_test.tolist(), ]
        y_test  = Y[ index_test.tolist() ]
        y_test = np.reshape(y_test,[len(y_test),1])    #BD

        # print(x_train.shape)

        # Normalizer
        from pytagi import Normalizer, Utils

        normalizer: Normalizer = Normalizer()

        x_mean, x_std = normalizer.compute_mean_std(x_train)
        y_mean, y_std = normalizer.compute_mean_std(y_train)

        # x_mean, x_std = normalizer.compute_mean_std(
        #     np.concatenate((x_train, x_test))
        # )
        # y_mean, y_std = normalizer.compute_mean_std(
        #     np.concatenate((y_train, y_test))
        # )


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

        print(data_loader["train"][0].shape)


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
        _, rmse_Epochlist, LL_Epochlist, _ = reg_task.train_UCI()

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

    # plot the mean RMSE and mean LL

    plt.plot(range(num_epochs), mean_RMSE)
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.show()

    plt.plot(range(num_epochs), mean_LL)
    plt.xlabel('Epochs')
    plt.ylabel('Log-likelihood')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(range(num_epochs), mean_RMSE)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('RMSE')
    ax[1].scatter(range(num_epochs), mean_LL)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Log-likelihood')
    # set the main title for the figure
    fig.suptitle(data_names[j])
    plt.savefig("results_small_UCI_TAGI_AGVI_Het/"+data_names[j]+"/RMSE_LL.png")

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
    with open(RESULTS_LL_learning_curve, "a") as file:
        file.write(str(mean_LL) + "\n")
    with open(RESULTS_RMSE_learning_curve, "a") as file:
        file.write(str(mean_RMSE) + "\n")




