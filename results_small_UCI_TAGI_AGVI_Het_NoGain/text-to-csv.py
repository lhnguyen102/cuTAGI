import pandas as pd

# Read the text file
with open("results_small_UCI_TAGI_AGVI_Het_NoGain/small_UCI_regression_Het_NoGain_table.txt", "r") as file:
    lines = file.readlines()

# Initialize lists to store data
datasets = []
rmse_values = []
log_likelihood_values = []
training_time_values = []

# Iterate through each line in the file
for line in lines[2:]:  # Skip the first two lines (header and separator)
    # Split the line into values
    values = line.strip().split("|")
    # Extract values for each column
    dataset = values[1].strip()
    rmse = values[2].strip()
    log_likelihood = values[3].strip()
    training_time = values[4].strip()
    # Append values to lists
    datasets.append(dataset)
    rmse_values.append(rmse)
    log_likelihood_values.append(log_likelihood)
    training_time_values.append(training_time)

# Create a DataFrame
data = {
    "Dataset": datasets,
    "RMSE": rmse_values,
    "Log-Likelihood": log_likelihood_values,
    "Average Training Time (in s.)": training_time_values
}
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv("results_small_UCI_TAGI_AGVI_Het_NoGain/Benchmark_table_TAGIV_NoGain.csv", index=False)

# Display the DataFrame
# print(df)
