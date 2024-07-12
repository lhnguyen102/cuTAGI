from typing import Union, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

from examples.data_loader import TimeSeriesDataloader


def main(num_epochs: int = 10, batch_size: int = 16, sigma_v: float = 2):
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 3
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 24  # for rolling window predictions in the test set
    early_stopping_criteria = "log_lik"  # 'log_lik' or 'mse'
    patience = 10

    # Network
    net = Sequential(
        LSTM(num_features, 40, input_seq_len),
        LSTM(40, 40, input_seq_len),
        LSTM(40, 40, input_seq_len),
        Linear(40 * input_seq_len, 1),
    )
    net.to_device("cuda")
    # net.set_threads(12)
    out_updater = OutputUpdater(net.device)

    # Loop over each time series in the benchmark
    nb_ts = 963
    ts_idx = np.arange(0, nb_ts)  # time series no.
    ytestPd = np.full((168, nb_ts), np.nan)
    SytestPd = np.full((168, nb_ts), np.nan)
    for ts in ts_idx:
        # options for early stopping
        log_lik_optim = -1e100
        mse_optim = 1e100
        epoch_optim = 1
        net_optim = []  # to save optimal net at the optimal epoch

        train_dtl = TimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_train.csv",
            date_time_file="data/traffic/traffic_2008_01_14_train_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            time_covariates=["hour_of_day", "day_of_week"],
        )
        val_dtl = TimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_val.csv",
            date_time_file="data/traffic/traffic_2008_01_14_val_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
            time_covariates=["hour_of_day", "day_of_week"],
        )
        test_dtl = TimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_test.csv",
            date_time_file="data/traffic/traffic_2008_01_14_test_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
            time_covariates=["hour_of_day", "day_of_week"],
        )

        # Viz
        viz = PredictionViz(task_name="forecasting", data_name="sin_signal")

        # -------------------------------------------------------------------------#
        # Training
        mses = []
        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            batch_iter = train_dtl.create_data_loader(batch_size, False)

            # Decaying observation's variance
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.3, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * len(output_col),), sigma_v**2, dtype=np.float32
            )
            for x, y in batch_iter:
                # Feed forward
                m_pred, _ = net(x)

                # Update output layer
                out_updater.update(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    var_obs=var_y,
                    delta_states=net.input_delta_z_buffer,
                )

                # Feed backward
                net.backward()
                net.step()

                # Training metric
                pred = normalizer.unstandardize(
                    m_pred, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
                )
                obs = normalizer.unstandardize(
                    y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
                )
                mse = metric.mse(pred, obs)
                mses.append(mse)

            # Validation
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds = []
            var_preds = []
            y_val = []
            x_val = []

            for x, y in val_batch_iter:
                # Predicion
                m_pred, v_pred = net(x)

                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v**2)
                x_val.extend(x)
                y_val.extend(y)

            mu_preds = np.array(mu_preds)
            std_preds = np.array(var_preds) ** 0.5
            y_val = np.array(y_val)
            x_val = np.array(x_val)

            mu_preds = normalizer.unstandardize(
                mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds = normalizer.unstandardize_std(
                std_preds, train_dtl.x_std[output_col]
            )

            y_val = normalizer.unstandardize(
                y_val, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            # Compute log-likelihood for validation set
            mse_val = metric.mse(mu_preds, y_val)
            log_lik_val = metric.log_likelihood(
                prediction=mu_preds, observation=y_val, std=std_preds
            )

            # Progress bar
            pbar.set_description(
                f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.3f}",
                refresh=True,
            )

            # early-stopping
            if early_stopping_criteria == "mse":
                if mse_val < mse_optim:
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net
            elif early_stopping_criteria == "log_lik":
                if log_lik_val > log_lik_optim:
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net
            if epoch - epoch_optim > patience:
                break

        # -------------------------------------------------------------------------#
        # Testing
        net = net_optim  # load optimal net
        # test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % (rolling_window)
            if RW_idx > 0:
                x[-RW_idx * num_features :: num_features] = mu_preds[-RW_idx:]
            #

            # Predicion
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)
            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        mu_preds = normalizer.unstandardize(
            mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )
        std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std[output_col])

        y_test = normalizer.unstandardize(
            y_test, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )

        # save test predicitons for each time series
        ytestPd[:, ts] = mu_preds.flatten()
        SytestPd[:, ts] = std_preds.flatten() ** 2

    # np.savetxt("traffic_2008_01_14_ytestPd_pyTAGI.csv", ytestPd, delimiter=",")
    # np.savetxt("traffic_2008_01_14_SytestPd_pyTAGI.csv", SytestPd, delimiter=",")
    # Compute log-likelihood
    # mse = metric.mse(mu_preds, y_test)
    # log_lik = metric.log_likelihood(
    #     prediction=mu_preds, observation=y_test, std=std_preds
    # )

    # ##Visualization
    # viz.plot_predictions(
    #     x_test=test_dtl.dataset["date_time"][: len(y_test)],
    #     y_test=y_test,
    #     y_pred=mu_preds,
    #     sy_pred=std_preds,
    #     std_factor=1,
    #     label="time_series_forecasting",
    #     title=r"\textbf{Time Series Forecasting}",
    #     time_series=True,
    # )

    # print("#############")
    # print(f"Val MSE           : {mse_optim: 0.2f}")
    # print(f"Val Log-likelihood: {log_lik_optim: 0.2f}")
    # print(f"Test MSE           : {mse: 0.2f}")
    # print(f"Test Log-likelihood: {log_lik: 0.2f}")


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
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        sy_pred: np.ndarray,
        std_factor: int,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        sy_test: Optional[np.ndarray] = None,
        label: str = "diag",
        title: Optional[str] = None,
        eq: Optional[str] = None,
        x_eq: Optional[float] = None,
        y_eq: Optional[float] = None,
        time_series: bool = False,
        save_folder: Optional[str] = None,
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


if __name__ == "__main__":
    fire.Fire(main)
