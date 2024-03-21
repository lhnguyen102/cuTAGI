import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


datasets = ["Boston_housing", "Concrete", "Energy", "Yacht", "Wine", "Kin8nm", "Power-plant", "Protein"] #, "Kin8nm", "Naval", "Power-plant", "Protein"

data = []
num_epochs = 100

for dataset in datasets:

    folder_path = os.path.join("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het_NoGain", dataset)
    rmse_file = os.path.join(folder_path, "learning_curve_RMSE.txt")
    ll_file = os.path.join(folder_path, "learning_curve_LL.txt")
    time_file = os.path.join(folder_path, "runtime_train.txt")

    with open(ll_file, "r") as f_LL:
        values = f_LL.read()
        mean_LL = [float(x) for x in values.replace('[', '').replace(']', '').replace('\n', '').split()]

    with open(rmse_file, "r") as f_rmse:
        values = f_rmse.read()
        mean_RMSE = [float(x) for x in values.replace('[', '').replace(']', '').replace('\n', '').split()]

    # Load optimal epoch value from the text file
    optimal_epoch_path = "/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het_earlystop/"+dataset+"/optimal_epoch.txt"
    with open(optimal_epoch_path, 'r') as file:
        optimal_epoch = int(file.readline().strip())

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(range(num_epochs), mean_RMSE)
    ax[0].axvline(x=optimal_epoch, color='r', linestyle='--')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('RMSE')
    ax[1].scatter(range(num_epochs), mean_LL)
    ax[1].axvline(x=optimal_epoch, color='r', linestyle='--')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Log-likelihood')
    # set the main title for the figure
    fig.suptitle(dataset)
    plt.savefig("results_small_UCI_TAGI_AGVI_Het_NoGain/"+dataset+"/RMSE_LL_with_opt_Epoch.png")