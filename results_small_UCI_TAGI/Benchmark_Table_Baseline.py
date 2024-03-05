import os
import pandas as pd
import matplotlib.pyplot as plt

datasets = ["Boston_housing", "Concrete", "Energy", "Yacht", "Wine"] #, "Kin8nm", "Naval", "Power-plant", "Protein"

data = []

for dataset in datasets:

    folder_path = os.path.join("/home/bd/projects/cuTAGI/results_small_UCI_TAGI", dataset)
    rmse_file = os.path.join(folder_path, "RMSEtest.txt")
    ll_file = os.path.join(folder_path, "LLtest.txt")
    time_file = os.path.join(folder_path, "runtime_train.txt")

    with open(rmse_file, "r") as f:
        rmse = float(f.readline().strip())

    with open(ll_file, "r") as f:
        ll = float(f.readline().strip())

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
plt.savefig("/home/bd/projects/cuTAGI/results_small_UCI_TAGI/dataset_metrics_table.png", bbox_inches='tight', dpi=300)



# df.to_csv("/home/bd/projects/cuTAGI/results_small_UCI_TAGI_AGVI_Het/dataset_metrics.csv", index=False)
