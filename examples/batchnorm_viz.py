import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


class BatchNormViz:
    def __init__(self):
        self.data = {}

    def update(self, data_dict: Dict[str, Tuple[list]], mode: str = "train"):
        """Store or update the BN statistics for a given layer and mode (train or test)."""
        if mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', but got {mode}")

        self.data[mode] = data_dict

    def plot_and_save_overlap(
        self, layer_name: str, filename: str = "bn_stats_overlap.png"
    ):
        """
        Plot a figure that shows both training and testing BN statistics
        on the same subplots for a given layer, and save it as a PNG file.

        BatchNorm statistics include:
            - mu_batch: mean of the batch
            - var_batch: variance of the batch
            - mu_ema_batch: mean of the eponech moving average (ema) of the batch
            - var_ema_batch: variance of the ema of the batch

        Figure contains two subplots side by side:
            - First subplot (axes[0]) displays the mean values and their standard deviations.
            - Second subplot (axes[1]) displays the variance values and their standard deviations.
        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for mode in ["train", "test"]:
            if mode not in self.data:
                continue
            if layer_name not in self.data[mode]:
                continue

            mu_batch, var_batch, mu_ema_batch, var_ema_batch = self.data[mode][
                layer_name
            ]
            mu_batch = np.array(mu_batch)
            var_batch = np.array(var_batch)
            mu_ema_batch = np.array(mu_ema_batch)
            var_ema_batch = np.array(var_ema_batch)

            mu_batch_mean = np.mean(mu_batch, axis=1)
            var_batch_mean = np.mean(var_batch, axis=1)
            mu_ema_batch_mean = np.mean(mu_ema_batch, axis=1)
            var_ema_batch_mean = np.mean(var_ema_batch, axis=1)

            mu_batch_std = np.std(mu_batch, axis=1)
            var_batch_std = np.std(var_batch, axis=1)
            mu_ema_batch_std = np.std(mu_ema_batch, axis=1)
            var_ema_batch_std = np.std(var_ema_batch, axis=1)

            x_indices = np.arange(len(mu_batch_mean))

            # Plot error bars for Mean
            axes[0].errorbar(
                x_indices,
                mu_batch_mean,
                yerr=mu_batch_std,
                label=f"{mode} mu_batch",
                fmt="o",
                capsize=3,
                alpha=0.7,
            )
            axes[0].errorbar(
                x_indices,
                mu_ema_batch_mean,
                yerr=mu_ema_batch_std,
                label=f"{mode} mu_ema",
                fmt="o",
                capsize=3,
                alpha=0.7,
            )

            # Plot error bars for Variance
            axes[1].errorbar(
                x_indices,
                var_batch_mean,
                yerr=var_batch_std,
                label=f"{mode} var_batch",
                fmt="o",
                capsize=3,
                alpha=0.7,
            )
            axes[1].errorbar(
                x_indices,
                var_ema_batch_mean,
                yerr=var_ema_batch_std,
                label=f"{mode} var_ema",
                fmt="o",
                capsize=3,
                alpha=0.7,
            )

        # Configure subplot for Mean
        axes[0].set_title(f"Mean (Layer: {layer_name})")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Mean Value")
        axes[0].legend()

        # Configure subplot for Variance
        axes[1].set_title(f"Variance (Layer: {layer_name})")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Variance Value")
        axes[1].legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)

        print(f"Plot for layer '{layer_name}' saved to '{filename}'.")

    def plot_all_layers(self, folder_name: str = "saved_results"):
        """Plot and save overlap figures for all batchnorm layers"""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        train_layers = set(self.data.get("train", {}).keys())
        test_layers = set(self.data.get("test", {}).keys())
        if train_layers != test_layers:
            raise ValueError(
                f"Layer names in train and test data are different: {train_layers} != {test_layers}"
            )

        # get layer name
        layer_names = self.data.get("test", {}).keys()
        for layer_name in layer_names:
            filename = f"{folder_name}/{layer_name}.png"
            self.plot_and_save_overlap(layer_name, filename=filename)
