import os
import pandas as pd
import matplotlib.pyplot as plt

datasets = ["Boston_housing", "Concrete", "Energy", "Yacht", "Wine", "Kin8nm", "Naval", "Power-plant", "Protein"] #, "Kin8nm", "Naval", "Power-plant", "Protein"

data = []

for dataset in datasets:

    folder_path = os.path.join("results_small_UCI_TAGI_AGVI_Het_NoGain", dataset)
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
with open("results_small_UCI_TAGI_AGVI_Het_NoGain/small_UCI_regression_Het_NoGain_table.txt", "w") as f:
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
plt.savefig("results_small_UCI_TAGI_AGVI_Het_NoGain/small_UCI_regression_Het_table_NoGain.png", bbox_inches='tight', dpi=300)



# df.to_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI/small_UCI_regression_table.csv", index=False)


### Getting the Combined Table

# # Read the CSV files into DataFrames
# df1 = pd.read_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI/dataset_metrics.csv")
# df2 = pd.read_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/dataset_metrics_Het.csv")

# # Combine the DataFrames
# combined_df = pd.merge(df1, df2, on="Dataset")
# combined_df = combined_df.round(3)

# combined_df.columns = ["Dataset",
#                        "RMSE (Baseline)", "Log-Likelihood (Baseline)", "Training Time (in s.) (Baseline)",
#                        "RMSE (TAGI-V)", "Log-Likelihood (TAGI-V)", "Training Time (in s.) (TAGI-V)"]


# # Save the combined DataFrame to a new CSV file
# combined_df.to_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/combined_dataset.csv", index=False)

# # Create a table visualization using matplotlib
# plt.figure(figsize=(15, 7))
# ax = plt.subplot(111, frame_on=False)  # Remove frame
# ax.xaxis.set_visible(False)  # Hide x-axis
# ax.yaxis.set_visible(False)  # Hide y-axis
# table = ax.table(cellText=combined_df.values,
#                  colLabels=combined_df.columns,
#                  loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.4, 1.4)  # Adjust size

# # Save the table as a PNG file
# plt.savefig("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/combined_table.png", bbox_inches='tight', pad_inches=0)
