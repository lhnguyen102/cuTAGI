# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from examples.data_loader import TimeSeriesDataloader
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from pytagi.nn import SLSTM, OutputUpdater, Sequential, SLinear


def main(num_epochs: int = 100, batch_size: int = 1, sigma_v: float = 1, ts_run: int = 81):
    # Dataset
    output_col = [0]
    num_features = 2
    input_seq_len = 20
    output_seq_len = 1
    seq_stride = 1

    data_file_train = "./data/hq/train100/split_train_values.csv"
    data_file_val = "./data/hq/split_val_values.csv"
    data_file_test = "./data/hq/split_test_values.csv"
    data_time_train = "./data/hq/train100/split_train_datetimes.csv"
    data_time_val = "./data/hq/split_val_datetimes.csv"
    data_time_test = "./data/hq/split_test_datetimes.csv"

    # cols= range(112)
    # cols= [1]
    cols = [ts_run] 
    df_train = pd.read_csv(data_file_train, skiprows=1, delimiter=",", header=None, usecols=cols)
    df_val = pd.read_csv(data_file_val, skiprows=1, delimiter=",", header=None, usecols=cols)
    df_test = pd.read_csv(data_file_test, skiprows=1, delimiter=",", header=None, usecols=cols)
    df_train_time = pd.read_csv(data_time_train, skiprows=1, delimiter=",", header=None, usecols=cols)
    df_val_time = pd.read_csv(data_time_val, skiprows=1, delimiter=",", header=None, usecols=cols)
    df_test_time = pd.read_csv(data_time_test, skiprows=1, delimiter=",", header=None, usecols=cols)

    # plt.plot(df_train.values)
    # plt.show()

    num_ts = df_train.shape[1]
    ts_list = np.random.permutation(num_ts)
    num_iter = int(np.ceil(num_ts/batch_size))
    time_covariates=["week_of_year"]
    mse_optim = 1e10
    epoch_optim = 0
    patience = 5
    infer_window_len = 80+52*2
    
    # # Data loader
    train_dtl_dict = {}
    val_dtl_dict = {}
    test_dtl_dict ={}

    for ts in range(num_ts):
        df_train_temp = df_train.iloc[:,[ts]]
        df_train_temp.index = pd.to_datetime(df_train_time.iloc[:, ts])
        last_idx = df_train_temp.iloc[:, 0].last_valid_index()
        df_train_temp = df_train_temp.loc[:last_idx]
        obs_infer = df_train_temp.values[:infer_window_len+input_seq_len].copy()
        df_train_temp.values[:infer_window_len+input_seq_len] = np.nan

        num_remove = 52 - input_seq_len
        df_val_temp = df_val.iloc[num_remove:,[ts]]
        df_val_temp.index = pd.to_datetime(df_val_time.iloc[num_remove:, ts])
        last_idx = df_val_temp.iloc[:, 0].last_valid_index()
        df_val_temp = df_val_temp.loc[:last_idx]

        df_test_temp = df_test.iloc[num_remove:,[ts]]
        df_test_temp.index = pd.to_datetime(df_test_time.iloc[num_remove:, ts])
        last_idx = df_test_temp.iloc[:, 0].last_valid_index()
        df_test_temp = df_test_temp.loc[:last_idx]

        train_dtl_dict[ts] = TimeSeriesDataloader(
            x_file="",
            date_time_file="",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            time_covariates =time_covariates,
            keep_last_time_cov=True,
            stride=seq_stride,
            df = df_train_temp,
        )

        val_dtl_dict[ts] = TimeSeriesDataloader(
            x_file="",
            date_time_file="",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            df = df_val_temp,
            x_mean=train_dtl_dict[ts].x_mean,
            x_std=train_dtl_dict[ts].x_std,
            time_covariates =time_covariates,
            keep_last_time_cov=True,
        )

        test_dtl_dict[ts] = TimeSeriesDataloader(
            x_file="",
            date_time_file="",

            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            df = df_test_temp,
            x_mean=train_dtl_dict[ts].x_mean,
            x_std=train_dtl_dict[ts].x_std,
            time_covariates =time_covariates,
            keep_last_time_cov=True,
        )

        obs_infer = normalizer.standardize(data=obs_infer, mu=train_dtl_dict[ts].x_mean[0], std=train_dtl_dict[ts].x_std[0])
    # Viz
    viz = PredictionViz(task_name="forecasting", data_name="sin_signal")

    # Network
    net = Sequential(
        SLSTM(num_features + input_seq_len - 1, 40, False, 1),
        SLSTM(40, 40, True, 1),
        SLinear(40, 1),
    )

    # net.to_device("cuda")
    net.set_threads(1)  # multi-processing is slow on a small net
    net.num_samples = train_dtl_dict[ts].dataset["value"][0].shape[0]
    out_updater = OutputUpdater(net.device)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    # Initialize the sequence length
    mu_sequence = np.ones(input_seq_len, dtype=np.float32)
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        batch_iter = train_dtl_dict[ts].create_data_loader(batch_size, shuffle=False)

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full(
            (batch_size * len(output_col),), sigma_v**2, dtype=np.float32
        )
        y_train = []

        for idx_sample, (x, y) in enumerate(batch_iter):

            # replace nan in input x by the lstm_prediction:
            if idx_sample < input_seq_len + infer_window_len:
                x = replace_with_prediction(x, mu_sequence)

            x = np.nan_to_num(x, nan=0.0)
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
                m_pred,
                train_dtl_dict[ts].x_mean[output_col],
                train_dtl_dict[ts].x_std[output_col],
            )
            y_train.append(y)
            obs = normalizer.unstandardize(
                y, train_dtl_dict[ts].x_mean[output_col], train_dtl_dict[ts].x_std[output_col]
            )

            mse = metric.mse(pred, obs)
            mses.append(mse)

            # Add new prediction to mu_sequence
            mu_sequence = np.append(mu_sequence, m_pred)
            mu_sequence = mu_sequence[-input_seq_len:]

        # Smoother
        lstm_states = net.get_lstm_states()
        mu_zo_smooth, var_zo_smooth = net.smoother()
        zo_smooth_std = np.array(var_zo_smooth) ** 0.5
        mu_sequence = np.ones(input_seq_len, dtype=np.float32)
        # mu_sequence = mu_zo_smooth[:input_seq_len]
        # mu_sequence = np.random.rand(input_seq_len)

        # Figures for each epoch for debugging
        # t = np.arange(len(mu_zo_smooth))
        # t_train = np.arange(len(y_train))
        # t_infer_len = np.arange(infer_window_len)
        # plt.figure()
        # plt.plot(t_train, y_train, color="r", label=r"$y_{true}$")
        # plt.plot(t, mu_zo_smooth, color="b", label=r"pred")
        # plt.plot(t_infer_len,obs_infer[input_seq_len:], color="r")
        # plt.fill_between(
        #     t,
        #     mu_zo_smooth - zo_smooth_std,
        #     mu_zo_smooth + zo_smooth_std,
        #     alpha=0.2,
        #     label="1 Std Dev",
        #     color="b",
        # )
        # plt.axvline(x=infer_window_len, color='k', linestyle='--')
        # plt.xlim(0, infer_window_len)
        # plt.legend()
        # filename = f"saved_results/hq_smoother#{epoch}.png"
        # plt.savefig(filename)
        # plt.close()

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {np.nansum(mses)/np.sum(~np.isnan(mses)):>7.2f}",
            refresh=True,
        )

    # Plot final smoothed values
    t = np.arange(len(mu_zo_smooth))
    t_train = np.arange(len(y_train))
    t_infer_len = np.arange(infer_window_len)
    plt.figure(figsize=(12, 8))
    plt.title("Smoothed SLSTM Output", fontsize=1.1 * 28, fontweight="bold")
    plt.plot(t_train, y_train, color="r", label=r"$y_{true}$")
    plt.plot(t, mu_zo_smooth, color="b", label=r"smooth")
    plt.plot(t_infer_len,obs_infer[input_seq_len:], color="r")
    plt.axvline(x=infer_window_len, color='k', linestyle='--')
    plt.fill_between(
        t,
        mu_zo_smooth - zo_smooth_std,
        mu_zo_smooth + zo_smooth_std,
        color="b",
        alpha=0.3,
        label="1 Std Dev",
    )
    plt.legend(
        loc="upper right",
        edgecolor="black",
        fontsize=28,
        ncol=1,
        framealpha=0.3,
        frameon=False,
    )
    plt.ylim(-3, 3)
    plt.tick_params(axis="both", which="both", direction="inout", labelsize=28)
    plt.legend()
    filename = f"saved_results/smooth_infer_hq_ts{ts_run}.png"
    plt.savefig(filename)
    plt.close()

    # -------------------------------------------------------------------------#
    # Testing
    test_batch_iter = val_dtl_dict[ts].create_data_loader(batch_size, shuffle=False)
    mu_preds = []
    var_preds = []
    y_test = []
    x_test = []

    net.set_lstm_states(lstm_states)
    for x, y in test_batch_iter:
        # Predicion
        x = np.nan_to_num(x, nan=0.0)
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
        mu_preds, train_dtl_dict[ts].x_mean[output_col], train_dtl_dict[ts].x_std[output_col]
    )
    std_preds = normalizer.unstandardize_std(
        std_preds, train_dtl_dict[ts].x_std[output_col]
    )

    y_test = normalizer.unstandardize(
        y_test, train_dtl_dict[ts].x_mean[output_col], train_dtl_dict[ts].x_std[output_col]
    )

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    # Visualization
    viz.plot_predictions(
        x_test=val_dtl_dict[ts].dataset["date_time"][: len(y_test)],
        y_test=y_test,
        y_pred=mu_preds,
        sy_pred=std_preds,
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
            axis="both",
            which="both",
            direction="inout",
            labelsize=self.fontsize,
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


def replace_with_prediction(x, mu_sequence):
    nan_indices = np.where(np.isnan(x))[0]
    x[nan_indices] = mu_sequence[nan_indices]
    return x


if __name__ == "__main__":
    fire.Fire(main)
