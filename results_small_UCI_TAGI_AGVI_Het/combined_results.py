import pandas as pd
import matplotlib.pyplot as plt



# Read data from both CSV files
df1 = pd.read_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI/dataset_metrics.csv")
df2 = pd.read_csv("/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/dataset_metrics_Het.csv")

# Plot the data
plt.figure(figsize=(10, 6))

# Plot RMSE
plt.subplot(3, 1, 1)
plt.bar(df1['Dataset'], df1['RMSE'], color='blue', alpha=0.5, label='Baseline')
plt.bar(df2['Dataset'], df2['RMSE'], color='red', alpha=0.5, label='TAGI-V')
plt.title('RMSE')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend()

# Plot Log-Likelihood
plt.subplot(3, 1, 2)
plt.bar(df1['Dataset'], df1['Log-Likelihood'], color='blue', alpha=0.5, label='Baseline')
plt.bar(df2['Dataset'], df2['Log-Likelihood'], color='red', alpha=0.5, label='TAGI-V')
plt.title('Log-Likelihood')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend()

# Plot Average Training Time
plt.subplot(3, 1, 3)
plt.bar(df1['Dataset'], df1['Average Training Time (in s.)'], color='blue', alpha=0.5, label='Baseline')
plt.bar(df2['Dataset'], df2['Average Training Time (in s.)'], color='red', alpha=0.5, label='TAGI-V')
plt.title('Average Training Time (in s.)')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as PNG
plt.savefig('/home/bd/Documents/cuTAGI/results_small_UCI_TAGI_AGVI_Het/comparison_plot.png')

# Show the plot (optional)
plt.show()