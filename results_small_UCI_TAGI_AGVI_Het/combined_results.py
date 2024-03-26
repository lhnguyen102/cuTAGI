import pandas as pd
import matplotlib.pyplot as plt

# Read data from baseline version
baseline_df = pd.read_csv("results_small_UCI_TAGI/Benchmark_table_Baseline.csv")

# Read data from TAGIV with modified gain version
tagiv_mod_df = pd.read_csv("results_small_UCI_TAGI_AGVI_Het/Benchmark_table_TAGIV.csv")

# Read data from TAGIV with gain = 1 version
tagiv_gain1_df = pd.read_csv("results_small_UCI_TAGI_AGVI_Het_NoGain/Benchmark_table_TAGIV_NoGain.csv")

def parse_mean_std(value):
    mean, stdv = value.split(' ± ')
    return float(mean)

# PARSE RMSE & LL
baseline_df['RMSE'] = baseline_df['RMSE'].apply(parse_mean_std)
tagiv_mod_df['RMSE'] = tagiv_mod_df['RMSE'].apply(parse_mean_std)
tagiv_gain1_df['RMSE'] = tagiv_gain1_df['RMSE'].apply(parse_mean_std)

baseline_df['Log-Likelihood'] = baseline_df['Log-Likelihood'].apply(parse_mean_std)
tagiv_mod_df['Log-Likelihood'] = tagiv_mod_df['Log-Likelihood'].apply(parse_mean_std)
tagiv_gain1_df['Log-Likelihood'] = tagiv_gain1_df['Log-Likelihood'].apply(parse_mean_std)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot RMSE
plt.subplot(3, 1, 1)
plt.bar(baseline_df['Dataset'], baseline_df['RMSE'], color='blue', alpha=0.5, label='Baseline')
plt.bar(tagiv_mod_df['Dataset'], tagiv_mod_df['RMSE'], color='red', alpha=0.5, label='TAGI-V')
plt.bar(tagiv_gain1_df['Dataset'], tagiv_gain1_df['RMSE'], color='green', alpha=0.5, label='TAGI-V (no gain)')
plt.title('RMSE')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend()

# Plot Log-Likelihood
plt.subplot(3, 1, 2)
plt.bar(baseline_df['Dataset'], baseline_df['Log-Likelihood'], color='blue', alpha=0.5, label='Baseline')
plt.bar(tagiv_mod_df['Dataset'], tagiv_mod_df['Log-Likelihood'], color='red', alpha=0.5, label='TAGI-V')
plt.bar(tagiv_gain1_df['Dataset'], tagiv_gain1_df['Log-Likelihood'], color='green', alpha=0.5, label='TAGI-V (no gain)')
plt.title('Log-Likelihood')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend()

# # Plot Average Training Time
# plt.subplot(3, 1, 3)
# plt.bar(baseline_df['Dataset'], baseline_df['Average Training Time (in s.)'], color='blue', alpha=0.5, label='Baseline')
# plt.bar(tagiv_mod_df['Dataset'], tagiv_mod_df['Average Training Time (in s.)'], color='red', alpha=0.5, label='TAGI-V')
# plt.title('Average Training Time (in s.)')
# plt.xlabel('Dataset')
# plt.ylabel('Value')
# plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as PNG
plt.savefig('results_small_UCI_TAGI_AGVI_Het/comparison_plot.png')
