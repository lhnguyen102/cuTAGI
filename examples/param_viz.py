import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ParameterDistributionVisualizer:
    """
    A class to track and visualize neural network parameter distributions across training updates.
    """

    def __init__(self):
        self.layer_updates_data = defaultdict(list)

    def record_params(self, state_dict: dict):
        """
        Extract mu_w, var_w, mu_b, var_b from net.state_dict()
        and append them as flattened arrays to 'updates_data'.
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

            self.layer_updates_data[layer_name].append(
                (mu_w_vals, var_w_vals, mu_b_vals, var_b_vals)
            )

    def _plot_distribution_3d(
        self,
        param_index: int,
        param_name: str,
        figsize: tuple,
        output_dir: str,
        elev: float = -120,
        azim: float = 120,
    ):
        """
        Helper function to plot 3D distribution for a specific parameter.
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for layer_name, updates_list in self.layer_updates_data.items():
            if not updates_list:
                continue

            # Filter out empty parameter arrays
            valid_params = [
                u[param_index] for u in updates_list if u[param_index].size > 0
            ]
            if not valid_params:
                print(
                    f"No valid data for {param_name} in layer {layer_name}. Skipping."
                )
                continue

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            # Concatenate parameter values across all updates to get global min/max
            all_param = np.concatenate([u[param_index] for u in updates_list])
            x_min, x_max = all_param.min(), all_param.max()
            x_grid = np.linspace(x_min, x_max, 100)

            for update_idx, update_vals in enumerate(updates_list):
                param_vals = update_vals[param_index]

                if param_vals.size == 0:
                    continue

                # Handle identical values edge case
                if np.all(param_vals == param_vals[0]):
                    print(
                        f"Identical values found for {param_name} in layer {layer_name}, update {update_idx}. Skipping KDE."
                    )
                    continue
                param_vals = update_vals[param_index]
                kde = gaussian_kde(param_vals)
                density = kde(x_grid)

                # For each update, we plot x (param_name) vs. z (density) in the plane,
                # and shift them along y = update_idx for a "waterfall" look.
                y_vals = np.full_like(x_grid, update_idx, dtype=float)

                line = ax.plot(
                    x_grid,  # x-axis: param_name
                    y_vals,  # y-axis: update index (the waterfall shift)
                    density,  # z-axis: density
                    label=f"Update {update_idx}",
                )
                line_color = line[0].get_color()

                # Fill the area under the curve in 3D
                verts = []
                # Add the curve vertices
                for x, z in zip(x_grid, density):
                    verts.append((x, update_idx, z))
                verts.append((x_grid[-1], update_idx, 0))
                verts.append((x_grid[0], update_idx, 0))

                poly = Poly3DCollection([verts], facecolors=line_color, alpha=0.3)
                ax.add_collection3d(poly)

            ax.set_xlabel(param_name)
            ax.set_ylabel("Update index")
            ax.set_zlabel("Density")

            num_updates = len(updates_list)
            ax.set_yticks(range(num_updates))

            # Remove grid lines for x and y axes
            ax.xaxis._axinfo["grid"].update({"linewidth": 0})
            ax.zaxis._axinfo["grid"].update({"linewidth": 0})

            ax.view_init(azim=azim)

            ax.set_title(f"Layer: {layer_name} - {param_name} distribution")
            plt.tight_layout()

            # Save if output directory is specified
            if output_dir is not None:
                filename = (
                    f"{layer_name.replace('.', '_')}_{param_name}_distribution.png"
                )
                save_path = os.path.join(output_dir, filename)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            plt.close(fig)

    def plot_distributions(self, figsize=(8, 6), output_dir: str = None):
        """
        Plots 3D distributions for mu_w, var_w, mu_b, and var_b.
        """
        self._plot_distribution_3d(0, "mu_w", figsize, output_dir)
        self._plot_distribution_3d(1, "var_w", figsize, output_dir)
        self._plot_distribution_3d(2, "mu_b", figsize, output_dir)
        self._plot_distribution_3d(3, "var_b", figsize, output_dir)

    def plot_initial_vs_final_differences(self, figsize=(8, 6), output_dir: str = None):
        """
        Plots the differences in mu_w, var_w, mu_b, and var_b between initial
        and final updates as histograms.
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for layer_name, updates_list in self.layer_updates_data.items():
            if len(updates_list) < 2:
                continue

            initial_update = updates_list[0]
            final_update = updates_list[-1]

            diffs = {
                "mu_w_diff": final_update[0] - initial_update[0],
                "var_w_diff": final_update[1] - initial_update[1],
                "mu_b_diff": (
                    final_update[2] - initial_update[2]
                    if initial_update[2].size > 0
                    else None
                ),
                "var_b_diff": (
                    final_update[3] - initial_update[3]
                    if initial_update[3].size > 0
                    else None
                ),
            }

            for param_name, diff_values in diffs.items():
                if diff_values is None or len(diff_values) == 0:
                    continue

                plt.figure(figsize=figsize)
                plt.hist(diff_values, bins=50, density=True, alpha=0.7)
                plt.title(f"Layer: {layer_name} - {param_name} differences")
                plt.xlabel(param_name)
                plt.ylabel("Density")
                plt.tight_layout()

                # Save if output directory is specified
                if output_dir is not None:
                    filename = (
                        f"{layer_name.replace('.', '_')}_{param_name}_differences.png"
                    )
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")

                plt.close()
