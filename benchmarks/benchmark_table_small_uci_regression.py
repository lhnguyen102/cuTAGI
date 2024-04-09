import os
import pandas as pd
import matplotlib.pyplot as plt

datasets = ["Boston_housing", "Concrete", "Energy", "Yacht", "Wine", "Kin8nm", "Naval", "Power-plant", "Protein"] #, "Kin8nm", "Naval", "Power-plant", "Protein"

data = []

for dataset in datasets:

    folder_path = os.path.join("benchmarks/logs/results_small_uci_regression_optimal_epoch", dataset)
    rmse_file = os.path.join(folder_path, "RMSEtest_opt_Epoch.txt")
    ll_file = os.path.join(folder_path, "LLtest_opt_Epoch.txt")
    time_file = os.path.join(folder_path, "runtime_train_opt_Epoch.txt")

    mean_values = []
    std_values = []
    with open(rmse_file, "r") as f:
        for line in f:
            mean, std = line.strip().split(" ± ")
            mean_values.append(float(mean))
            std_values.append(float(std))
        mean_average = sum(mean_values) / len(mean_values)
        std_average = sum(std_values) / len(std_values)
        rmse = "{:.3f} ± {:.3f}".format(mean_average, std_average)

    mean_values = []
    std_values = []
    with open(ll_file, "r") as f:
        for line in f:
            mean, std = line.strip().split(" ± ")
            mean_values.append(float(mean))
            std_values.append(float(std))
        mean_average = sum(mean_values) / len(mean_values)
        std_average = sum(std_values) / len(std_values)
        ll = "{:.3f} ± {:.3f}".format(mean_average, std_average)

    mean_values = []
    std_values = []
    with open(time_file, "r") as f:
        for line in f:
            mean, std = line.strip().split(" ± ")
            mean_values.append(float(mean))
            std_values.append(float(std))
        mean_average = sum(mean_values) / len(mean_values)
        std_average = sum(std_values) / len(std_values)
        time = "{:.3f} ± {:.3f}".format(mean_average, std_average)

    data.append([dataset, rmse, ll, time])

# Creating DataFrame
df = pd.DataFrame(data, columns=["Dataset", "RMSE", "Log-Likelihood", "Average Training Time (in s.)"])

# upto 3 decimal places
df_rounded = df.round(3)

# saving table as a markdown file (.txt)
with open("benchmarks/logs/results_small_uci_regression_optimal_epoch/small_uci_regression_table.txt", "w") as f:
    f.write(df_rounded.to_markdown(index=False))

# Plotting the rounded DataFrame as a table
plt.figure(figsize=(10, 6))
plt.table(cellText=df_rounded.values,
          colLabels=df_rounded.columns,
          cellLoc='center',
          loc='center')
plt.axis('off')  # Hide the axes
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("benchmarks/logs/results_small_uci_regression_optimal_epoch/small_uci_regression_table.png", bbox_inches='tight', dpi=300)



