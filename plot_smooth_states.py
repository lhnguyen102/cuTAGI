import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the raw CSV (no header) ---
# Make sure your CSV is named 'state_values.csv' and is in the same folder as this script.
df_raw = pd.read_csv("linear_state_summary.csv", header=None, skiprows=1)

# --- 2. Prepare time‐axis ---
# First two columns are [StateType, Variable], the rest are the values at each time step.
n_time = df_raw.shape[1] - 2
time_steps = list(range(1, n_time + 1))

# --- 3. Filter for only the 'mu' variables (priors, posts, smooths) ---
# Column 1 holds the variable name (e.g. 'mu_zo_priors', 'mu_zo_posts', 'mu_zo_smooths')
mu_df = df_raw[df_raw[1].str.startswith("mu")]

# --- 4. Plot ---
plt.figure(figsize=(8, 4))
cmap = plt.get_cmap('winter')
colors = cmap(np.linspace(0, 1, len(mu_df)))

for color, (_, row) in zip(colors, mu_df.iterrows()):
    label = row[1]
    values = row.iloc[2:].astype(float)
    print(len(values), len(time_steps), label)
    plt.plot(time_steps, values, color=color, label=label)

plt.axvline(x=24, color='red', linestyle='--', linewidth=1)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Evolution of μ Priors, Posteriors and Smoothed Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
