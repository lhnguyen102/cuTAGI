import os
from collections import defaultdict

import numpy as np
import wandb


class WandBLogger:

    def __init__(self, project_name, config=None):
        """Initialize the WandB Logger."""
        self.project_name = project_name
        self.config = config or {}
        self.run = None

    def init(self):
        """
        Initialize the WandB run, skipping if the API key is not set.
        """
        if os.getenv("WANDB_API_KEY"):
            self.run = wandb.init(project=self.project_name, config=self.config)
        else:
            print("WandB API key not found. Skipping WandB initialization.")

    def log(self, data):
        """Log data to WandB."""
        if self.run is not None:
            wandb.log(data)

    def finish(self):
        """Finish the WandB run."""
        if self.run is not None:
            self.run.finish()


class ParamStatTable:

    def __init__(self):
        # Each layer will store only the current and previous states
        self.layer_updates_data = defaultdict(lambda: [None, None])

    def record_params(self, state_dict: dict):
        """
        Extract mu_w, var_w, mu_b, var_b from state_dict
        and store only the current and previous data for each layer.
        """
        for layer_name, value in state_dict.items():
            mu_w_vals = np.array(value[0])
            var_w_vals = np.array(value[1])

            if len(value[2]) != 0 and len(value[3]) != 0:
                mu_b_vals = np.array(value[2])
                var_b_vals = np.array(value[3])
            else:
                mu_b_vals = np.array([])
                var_b_vals = np.array([])

            previous_data = self.layer_updates_data[layer_name][1]
            self.layer_updates_data[layer_name][0] = previous_data
            self.layer_updates_data[layer_name][1] = (
                mu_w_vals,
                var_w_vals,
                mu_b_vals,
                var_b_vals,
            )

    def _gather_row(
        self, layer_name: str, snapshots: list, use_diff_metric: bool
    ):
        """
        Helper method to compute mean, std, and difference metric for a layer.
        """
        if len(snapshots) == 0:
            return None

        # please handle the none case
        if snapshots[0] is None:
            mu_w_init, var_w_init, mu_b_init, var_b_init = 0, 0, 0, 0
            mu_w_current, var_w_current, mu_b_current, var_b_current = (
                snapshots[-1]
            )
        else:
            mu_w_init, var_w_init, mu_b_init, var_b_init = snapshots[0]
            mu_w_current, var_w_current, mu_b_current, var_b_current = (
                snapshots[-1]
            )

        # Compute std from variance
        var_w_init = np.sqrt(var_w_init)
        var_w_current = np.sqrt(var_w_current)
        var_b_init = np.sqrt(var_b_init)
        var_b_current = np.sqrt(var_b_current)

        if len(snapshots) > 1:
            d_mu_w = mu_w_current - mu_w_init
            d_var_w = var_w_current - var_w_init
            d_mu_b = mu_b_current - mu_b_init
            d_var_b = var_b_current - var_b_init
        else:
            d_mu_w = mu_w_current
            d_var_w = var_w_current
            d_mu_b = mu_b_current
            d_var_b = var_b_current

        diff_metric = (
            np.sum(np.abs(d_mu_w)) + np.sum(np.abs(d_var_w))
            if use_diff_metric
            else 0.0
        )

        # Select data to use based on use_diff_metric
        target_mu_w = d_mu_w if use_diff_metric else mu_w_current
        target_var_w = d_var_w if use_diff_metric else var_w_current
        target_mu_b = d_mu_b if use_diff_metric else mu_b_current
        target_var_b = d_var_b if use_diff_metric else var_b_current

        row = (
            layer_name,
            target_mu_w.mean() if target_mu_w.size > 0 else float("nan"),
            target_mu_w.std() if target_mu_w.size > 0 else float("nan"),
            target_var_w.mean() if target_var_w.size > 0 else float("nan"),
            target_var_w.std() if target_var_w.size > 0 else float("nan"),
            target_mu_b.mean() if target_mu_b.size > 0 else float("nan"),
            target_mu_b.std() if target_mu_b.size > 0 else float("nan"),
            target_var_b.mean() if target_var_b.size > 0 else float("nan"),
            target_var_b.std() if target_var_b.size > 0 else float("nan"),
            diff_metric,
        )

        return row

    def rows_to_dict(self):
        """
        Convert all rows to a dictionary format for logging with WandB.
        """
        logs = {}
        for layer_name, snapshots in self.layer_updates_data.items():
            row = self._gather_row(layer_name, snapshots, use_diff_metric=False)
            if row:
                (
                    layer_name,
                    target_mu_w_mean,
                    target_mu_w_std,
                    target_var_w_mean,
                    target_var_w_std,
                    target_mu_b_mean,
                    target_mu_b_std,
                    target_var_b_mean,
                    target_var_b_std,
                    diff_metric,
                ) = row
                logs.update(
                    {
                        f"{layer_name}/target_mu_w_mean": target_mu_w_mean,
                        f"{layer_name}/target_mu_w_std": target_mu_w_std,
                        f"{layer_name}/target_var_w_mean": target_var_w_mean,
                        f"{layer_name}/target_var_w_std": target_var_w_std,
                        f"{layer_name}/target_mu_b_mean": target_mu_b_mean,
                        f"{layer_name}/target_mu_b_std": target_mu_b_std,
                        f"{layer_name}/target_var_b_mean": target_var_b_mean,
                        f"{layer_name}/target_var_b_std": target_var_b_std,
                        f"{layer_name}/diff_metric": diff_metric,
                    }
                )
        return logs

    def print_parameter_distributions(self, topN: int = None):
        """
        Print a table of (mean, std) for each layer's mu_w, var_w, mu_b, var_b,
        using self.layer_updates_data.
        """
        rows = []

        for layer_name, snapshots in self.layer_updates_data.items():
            row = self._gather_row(layer_name, snapshots, use_diff_metric=True)
            if row:
                rows.append(row)

        if topN is not None:
            rows.sort(key=lambda x: x[-1], reverse=True)
            rows = rows[:topN]

        # Print header
        header = (
            f"{'Layer':<30} | {'d_mu_w_mean':>10} | {'d_mu_w_std':>10} | "
            f"{'d_var_w_mean':>10} | {'d_var_w_std':>10} | {'d_mu_b_mean':>10} | "
            f"{'d_mu_b_std':>10} | {'d_var_b_mean':>10} | {'d_var_b_std':>10}"
        )
        print(header)
        print("-" * len(header))

        for row in rows:
            print(
                f"{row[0]:<30} | "
                f"{row[1]:10.6f} | {row[2]:10.6f} | "
                f"{row[3]:10.6f} | {row[4]:10.6f} | "
                f"{row[5]:10.6f} | {row[6]:10.6f} | "
                f"{row[7]:10.6f} | {row[8]:10.6f}"
            )

    def print_current_parameter_distributions(self, topN: int = None):
        """
        Print a table of (mean, std) for the CURRENT values of mu_w, var_w, mu_b, var_b,
        but rank the layers by the same difference metric (init vs. current).
        """
        rows = []

        for layer_name, snapshots in self.layer_updates_data.items():
            row = self._gather_row(layer_name, snapshots, use_diff_metric=False)
            if row:
                rows.append(row)

        if topN is not None:
            rows.sort(key=lambda x: x[-1], reverse=True)
            rows = rows[:topN]

        header = (
            f"{'Layer':<30} | {'mu_w_mean':>10} | {'mu_w_std':>10} | "
            f"{'std_w_mean':>10} | {'std_w_std':>10} | {'mu_b_mean':>10} | "
            f"{'mu_b_std':>10} | {'std_b_mean':>10} | {'std_b_std':>10}"
        )
        print(header)
        print("-" * len(header))

        for row in rows:
            print(
                f"{row[0]:<30} | "
                f"{row[1]:10.6f} | {row[2]:10.6f} | "
                f"{row[3]:10.6f} | {row[4]:10.6f} | "
                f"{row[5]:10.6f} | {row[6]:10.6f} | "
                f"{row[7]:10.6f} | {row[8]:10.6f}"
            )
