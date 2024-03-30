import cProfile
import pstats
from io import StringIO
from typing import List, Union

import fire
import matplotlib.pyplot as plt
import memory_profiler
import numpy as np
from data_loader import TimeSeriesDataloader
from linear import Linear
from lstm import LSTM
from output_updater import OutputUpdater
from sequential import Sequential
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
import pandas as pd


class TimeSeriesForecaster:
    """Time series forecaster using TAGI"""

    def __init__(
        self,
        num_epochs: int,
        data_loader: dict,
        input_seq_len: int,
        output_seq_len: int,
        batch_size: int,
        output_col: List[int],
        sigma_v: float,
        viz: None,
        dtype=np.float32,
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.batch_size = batch_size
        self.output_col = output_col
        self.sigma_v = sigma_v

        self.network = Sequential(
            LSTM(1, 5, input_seq_len),
            LSTM(5, 5, input_seq_len),
            Linear(5 * input_seq_len, 1),
        )

        self.viz = viz
        self.dtype = dtype

    def train(self) -> None:
        """Train LSTM network"""
        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.network.device)

        # Inputs
        batch_size = self.batch_size

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs), desc="Training Progress")

        for epoch in pbar:
            # Decaying observation's variance
            self.sigma_v = exponential_scheduler(
                curr_v=self.sigma_v,
                min_v=0.3,
                decaying_factor=0.95,
                curr_iter=epoch,
            )
            var_obs = (
                np.zeros(
                    (batch_size, len(self.output_col) * self.output_seq_len),
                    dtype=np.float32,
                )
                + self.sigma_v**2
            )
            mses = []
            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]

                # Feed forward
                self.network(x_batch.flatten())

                # Update output layer
                output_updater.update(
                    output_states=self.network.output_z_buffer,
                    mu_obs=y_batch.flatten(),
                    var_obs=var_obs.flatten(),
                    delta_states=self.network.input_delta_z_buffer,
                )

                # Feed backward
                self.network.backward()
                self.network.step()

                # Loss
                mu_pred, _ = self.network.get_outputs()
                pred = normalizer.unstandardize(
                    norm_data=mu_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                mse = metric.mse(pred, obs.flatten())
                mses.append(mse)

            # Progress bar
            pbar.set_description(
                f"Epoch {epoch + 1}/{self.num_epochs}| mse: {sum(mses)/len(mses):>7.2f}",
                refresh=True,
            )

    def predict(self) -> None:
        """Make prediction for time series using TAGI"""

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network(x_batch.flatten())
            mu_pred, var_pred = self.network.get_outputs()

            mean_predictions.append(mu_pred)
            variance_predictions.append(var_pred + self.sigma_v**2)
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten()) ** 0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"]
        )

        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        # Visualization
        if self.viz is not None:
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=self.data_loader["datetime_test"][: len(y_test)],
                y_test=y_test,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=1,
                label="time_series_forecasting",
                title=r"\textbf{Time Series Forecasting}",
                time_series=True,
            )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")


class PredictionViz:
    """Visualization of prediction
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as Boston housing or toy example
        figsize: Size of figure
        fontsize: Font size for letter in the figure
        lw: linewidth
        ms: Marker size
        ndiv_x: Number of divisions for x-direction
        ndiv_y: Number of division for y-direciton
    """

    def __init__(
        self,
        task_name: str,
        data_name: str,
        figsize: tuple = (12, 12),
        fontsize: int = 28,
        lw: int = 3,
        ms: int = 10,
        ndiv_x: int = 4,
        ndiv_y: int = 4,
    ) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.figsize = figsize
        self.fontsize = fontsize
        self.lw = lw
        self.ms = ms
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def load_dataset(self, file_path: str, header: bool = False) -> np.ndarray:
        """Load dataset (*.csv)
        Args:
            file_path: File path to the data file
            header: Ignore hearder ?

        """

        # Load image data from *.csv file
        if header:
            df = pd.read_csv(file_path, skiprows=1, delimiter=",", header=None)
        else:
            df = pd.read_csv(file_path, skiprows=0, delimiter=",", header=None)

        return df[0].values

    def plot_predictions(
        self,
        x_train: Union[np.ndarray, None],
        y_train: Union[np.ndarray, None],
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        sy_pred: np.ndarray,
        std_factor: int,
        sy_test: Union[np.ndarray, None] = None,
        label: str = "diag",
        title: Union[str, None] = None,
        eq: Union[str, None] = None,
        x_eq: Union[float, None] = None,
        y_eq: Union[float, None] = None,
        time_series: bool = False,
        save_folder: Union[str, None] = None,
    ) -> None:
        """Compare prediciton distribution with theorical distribution

        x_train: Input train data
        y_train: Output train data
        x_test: Input test data
        y_test: Output test data
        y_pred: Prediciton of network
        sy_pred: Standard deviation of the prediction
        std_factor: Standard deviation factor
        sy_test: Output test's theorical standard deviation
        label: Name of file
        title: Figure title
        eq: Math equation for data
        x_eq: x-coordinate for eq
        y_eq: y-coordinate for eq

        """

        # Get max and min values
        if sy_test is not None:
            std_y = max(sy_test)
        else:
            std_y = 0

        if x_train is not None:
            max_y = np.maximum(max(y_test), max(y_train)) + std_y
            min_y = np.minimum(min(y_test), min(y_train)) - std_y
            max_x = np.maximum(max(x_test), max(x_train))
            min_x = np.minimum(min(x_test), min(x_train))
        else:
            max_y = max(y_test) + std_y
            min_y = min(y_test) - std_y
            max_x = max(x_test)
            min_x = min(x_test)

        # Plot figure
        plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set_title(title, fontsize=1.1 * self.fontsize, fontweight="bold")
        if eq is not None:
            ax.text(x_eq, y_eq, eq, color="k", fontsize=self.fontsize)
        ax.plot(x_test, y_pred, "r", lw=self.lw, label=r"$\mathbb{E}[Y^{'}]$")
        ax.plot(x_test, y_test, "k", lw=self.lw, label=r"$y_{true}$")

        ax.fill_between(
            x_test,
            y_pred - std_factor * sy_pred,
            y_pred + std_factor * sy_pred,
            facecolor="red",
            alpha=0.3,
            label=r"$\mathbb{{E}}[Y^{{'}}]\pm{}\sigma$".format(std_factor),
        )
        if sy_test is not None:
            ax.fill_between(
                x_test,
                y_test - std_factor * sy_test,
                y_test + std_factor * sy_test,
                facecolor="blue",
                alpha=0.3,
                label=r"$y_{{test}}\pm{}\sigma$".format(std_factor),
            )
        if x_train is not None:
            if time_series:
                marker = ""
                line_style = "-"
            else:
                marker = "o"
                line_style = ""
            ax.plot(
                x_train,
                y_train,
                "b",
                marker=marker,
                mfc="none",
                lw=self.lw,
                ms=0.2 * self.ms,
                linestyle=line_style,
                label=r"$y_{train}$",
            )

        ax.set_xlabel(r"$x$", fontsize=self.fontsize)
        ax.set_ylabel(r"$y$", fontsize=self.fontsize)
        if time_series:
            x_ticks = pd.date_range(min_x, max_x, periods=self.ndiv_x).values
        else:
            x_ticks = np.linspace(min_x, max_x, self.ndiv_x)
        y_ticks = np.linspace(min_y, max_y, self.ndiv_y)
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        ax.tick_params(
            axis="both", which="both", direction="inout", labelsize=self.fontsize
        )
        ax.legend(
            loc="upper right",
            edgecolor="black",
            fontsize=1 * self.fontsize,
            ncol=1,
            framealpha=0.3,
            frameon=False,
        )
        ax.set_ylim([min_y, max_y])
        ax.set_xlim([min_x, max_x])

        # Save figure
        if save_folder is None:
            plt.show()
        else:
            saving_path = f"saved_results/pred_{label}_{self.data_name}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()


# @memory_profiler.profile
def clsf_runner():
    """Run classification training"""
    # User-input
    num_epochs = 20
    output_col = [0]
    num_features = 1
    input_seq_len = 5
    output_seq_len = 1
    seq_stride = 1
    batch_size = 10
    sigma_v = 1.0
    x_train_file = "../../data/toy_time_series/x_train_sin_data.csv"
    datetime_train_file = "../../data/toy_time_series/train_sin_datetime.csv"
    x_test_file = "../../data/toy_time_series/x_test_sin_data.csv"
    datetime_test_file = "../../data/toy_time_series/test_sin_datetime.csv"

    # Data loader
    ts_data_loader = TimeSeriesDataloader(
        batch_size=batch_size,
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
    )
    data_loader = ts_data_loader.process_data(
        x_train_file=x_train_file,
        datetime_train_file=datetime_train_file,
        x_test_file=x_test_file,
        datetime_test_file=datetime_test_file,
    )

    # Visualzier
    viz = PredictionViz(task_name="forecasting", data_name="sin_signal")
    forecaster = TimeSeriesForecaster(
        num_epochs=num_epochs,
        data_loader=data_loader,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        output_col=output_col,
        batch_size=batch_size,
        sigma_v=sigma_v,
        viz=viz,
    )
    forecaster.train()
    forecaster.predict()


def memory_profiling_main():
    clsf_runner()


def profiler():
    """Run profiler"""
    pr = cProfile.Profile()
    pr.enable()

    # Run the main function
    memory_profiling_main()

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("time")
    ps.print_stats(20)  # Print only the top 20 functions

    # Print cProfile output to console
    print("Top 20 time-consuming functions:")
    print(s.getvalue())


def main(profile: bool = False):
    """Test API"""
    if profile:
        print("Profile training")
        profiler()
    else:
        clsf_runner()


if __name__ == "__main__":
    fire.Fire(main)
