import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


datasets = ["Boston_housing", "Concrete", "Energy", "Yacht", "Wine", "Kin8nm", "Naval", "Power-plant", "Protein"] #, "Kin8nm", "Naval", "Power-plant", "Protein"

data = []

for dataset in datasets:

    folder_path = os.path.join("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het", dataset)
    rmse_file = os.path.join(folder_path, "learning_curve_RMSE.txt")
    ll_file = os.path.join(folder_path, "learning_curve_LL.txt")
    time_file = os.path.join(folder_path, "runtime_train.txt")

    with open(ll_file, "r") as f:
        values = f.read()
        data_list = [float(x) for x in values.replace('[', '').replace(']', '').replace('\n', '').split()]
        ll = max(data_list)
        max_ll_index = data_list.index(ll)

        with open(rmse_file, "r") as f_rmse:
            values_rmse = f_rmse.read()
            data_list_rmse = [float(x) for x in values_rmse.replace('[', '').replace(']', '').replace('\n', '').split()]
            rmse = data_list_rmse[max_ll_index]




    with open(time_file, "r") as f:
        time = float(f.readline().strip())

    data.append([dataset, rmse, ll, time])

# Creating DataFrame
df = pd.DataFrame(data, columns=["Dataset", "RMSE", "Log-Likelihood", "Average Training Time (in s.)"])

df_rounded = df.round(3)

# Plotting the rounded DataFrame as a table
plt.figure(figsize=(10, 6))
plt.table(cellText=df_rounded.values,
          colLabels=df_rounded.columns,
          cellLoc='center',
          loc='center')
plt.axis('off')  # Hide the axes
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/dataset_metrics_table_Het.png", bbox_inches='tight', dpi=300)

df.to_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/dataset_metrics_Het.csv", index=False)


