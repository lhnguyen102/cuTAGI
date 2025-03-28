# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential
from pytagi import manual_seed

from examples.data_loader import TimeSeriesDataloader

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

import matplotlib as mpl

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
    }
)

# set line width to 1
mpl.rcParams["lines.linewidth"] = 1


def generate_changing_amplitude_sine(
    frequency=1, phase=0, sampling_rate=100, duration=10, change_points=None
):
    """
    Generate a sine wave time series with variable amplitude and frequency,
    ensuring continuity at changepoints by adjusting the phase.

    If `change_points` is None, a constant amplitude and frequency are used.
    Otherwise, the amplitude and frequency change at the specified time points,
    and the phase is updated to keep the sine wave continuous at each changepoint.

    Parameters
    ----------
    frequency : float, optional
        Default frequency of the sine wave (default is 1). This is used if a change point
        does not specify a frequency.
    phase : float, optional
        Initial phase in radians (default is 0).
    sampling_rate : int, optional
        Number of samples per second (default is 100).
    duration : int or float, optional
        Duration of the signal.
    change_points : list of tuple, optional
        Each tuple should specify (time, amplitude) or (time, amplitude, frequency).
        The amplitude and frequency change at these time points.

    Returns
    -------
    tuple
        t : ndarray
            Time points.
        y : ndarray
            Sine wave values.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    if change_points is None:
        y = np.sin(2 * np.pi * frequency * t + phase)
    else:
        y = np.zeros_like(t)
        # Initialize with the default frequency and phase for the first segment
        current_phase = phase
        current_freq = frequency

        # Process each segment defined by change_points
        for i in range(len(change_points) - 1):
            cp = change_points[i]
            start_time = cp[0]
            amplitude = cp[1]
            seg_freq = cp[2] if len(cp) > 2 else frequency

            # For segments after the first, adjust phase to ensure continuity
            if i > 0:
                # t_c is the current changepoint time
                t_c = start_time
                # Adjust phase so that:
                # sin(2*pi*seg_freq*t_c + new_phase) = sin(2*pi*current_freq*t_c + current_phase)
                current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                    2 * np.pi * seg_freq * t_c
                )
                current_freq = seg_freq

            # Determine end time for this segment
            next_cp = change_points[i + 1]
            end_time = next_cp[0]
            mask = (t >= start_time) & (t < end_time)
            y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)

        # Handle the final segment
        last_cp = change_points[-1]
        start_time = last_cp[0]
        amplitude = last_cp[1]
        seg_freq = last_cp[2] if len(last_cp) > 2 else frequency
        if len(change_points) > 1:
            t_c = start_time
            current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                2 * np.pi * seg_freq * t_c
            )
        mask = t >= start_time
        y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)
    return t, y


manual_seed(42)


def main(
    num_epochs: int = 20,
    batch_size: int = 1,
    sigma_v: float = 0.1,
    t_reset: Optional[int] = -1,
    plot_title: str = "",
):
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 1
    input_seq_len = 1
    output_seq_len = 1
    seq_stride = 1

    # Generate synthetic data
    frequency = 1 / 24  # One cycle per 24 hours
    phase = 0  # Initial phase
    sampling_rate = 1  # 1 sample per hour
    duration = 1 / frequency * 12  # Total duration

    t, y = generate_changing_amplitude_sine(
        frequency=frequency,
        phase=phase,
        sampling_rate=sampling_rate,
        duration=duration,
    )
    # Convert time array to datetime using a base date and slice out the last two cycles
    base_date = pd.Timestamp("2020-01-01")
    t_dates = pd.to_datetime(t, unit="h", origin=base_date)
    num_samples = int(2 * 24 * sampling_rate)
    t_test = t_dates[-(num_samples + 1) :]
    y_test = y[-(num_samples + 1) :]
    t_train = t_dates[:-num_samples]
    y_train = y[:-num_samples]

    # if directory does not exist, create it
    if not os.path.exists("data/toy_getLSTM_states"):
        os.makedirs("data/toy_getLSTM_states")

    # Save the data with datetime as strings using DataFrames
    df_train_x = pd.DataFrame(y_train, columns=["value"])
    df_train_dates = pd.DataFrame(t_train.astype(str), columns=["datetime"])
    df_train_x.to_csv("data/toy_getLSTM_states/x_train_sin_data.csv", index=False)
    df_train_dates.to_csv("data/toy_getLSTM_states/train_sin_datetime.csv", index=False)

    df_test_x = pd.DataFrame(y_test, columns=["value"])
    df_test_dates = pd.DataFrame(t_test.astype(str), columns=["datetime"])
    df_test_x.to_csv("data/toy_getLSTM_states/x_test_sin_data.csv", index=False)
    df_test_dates.to_csv("data/toy_getLSTM_states/test_sin_datetime.csv", index=False)

    train_dtl = TimeSeriesDataloader(
        x_file="data/toy_getLSTM_states/x_train_sin_data.csv",
        date_time_file="data/toy_getLSTM_states/train_sin_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
    )
    test_dtl = TimeSeriesDataloader(
        x_file="data/toy_getLSTM_states/x_test_sin_data.csv",
        date_time_file="data/toy_getLSTM_states/test_sin_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
    )

    # Network
    net = Sequential(
        LSTM(1, 40, input_seq_len),
        LSTM(40, 40, input_seq_len),
        Linear(40 * input_seq_len, 1),
    )

    net.to_device("cuda")
    # net.set_threads(1)  # multi-processing is slow on a small net
    out_updater = OutputUpdater(net.device)

    # Training
    mses = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size, False)

        # Set observation noise
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        states = []
        mu_preds_train = []
        var_preds_train = []

        for x, y in batch_iter:
            # Feed forward
            m_pred, S_pred = net(x)

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

            # get cell states
            states_dict = net.get_lstm_states()
            states.append(states_dict)

            # Training metric
            pred = normalizer.unstandardize(
                m_pred, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            obs = normalizer.unstandardize(
                y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            S_pred = normalizer.unstandardize_std(S_pred, train_dtl.x_std[output_col])
            mse = metric.mse(pred, obs)
            mses.append(mse)

            # Save predictions
            mu_preds_train.extend(pred)
            var_preds_train.extend(S_pred)

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.2f}",
            refresh=True,
        )

    # set states
    if t_reset is not None:
        net.set_lstm_states(states[t_reset])

    # Testing
    test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)
    mu_preds = []
    var_preds = []
    y_tests = []
    x_test = []

    for RW_idx_, (x, y) in enumerate(test_batch_iter):

        if RW_idx_ == 0:
            m_pred, v_pred = net(x)
        else:
            m_pred, v_pred = net(m_pred)

        mu_preds.extend(m_pred)
        var_preds.extend(v_pred + sigma_v**2)
        x_test.extend(x)
        y_tests.extend(y)

    mu_preds = np.array(mu_preds)
    std_preds = np.array(var_preds) ** 0.5
    y_tests = np.array(y_tests)
    x_test = np.array(x_test)

    mu_preds = normalizer.unstandardize(
        mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
    )
    std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std[output_col])

    y_tests = normalizer.unstandardize(
        y_tests, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
    )

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_tests)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_tests, std=std_preds
    )

    print("#############")
    print(f"MSE           : {mse: 0.2f}")
    print(f"Log-likelihood: {log_lik: 0.2f}")

    #  Plot
    plt.figure(figsize=(6, 2))
    plt.axvspan(
        t_train[0], t_train[-1], facecolor="dodgerblue", alpha=0.2, label="Training"
    )
    plt.plot(
        t_train,
        y_train,
        color="r",
        label=r"$y_{true}$",
        linewidth=1,
    )
    plt.plot(
        t_test[1:],
        y_test[1:],
        color="r",
        linewidth=1,
    )
    plt.plot(t_test[1:], mu_preds, color="b", label=r"$\mathbb{E}[Y']$")
    plt.fill_between(
        t_test[1:],
        mu_preds - std_preds,
        mu_preds + std_preds,
        facecolor="blue",
        alpha=0.3,
        label=r"$\mathbb{E}[Y'] \pm \sigma$",
    )
    plt.plot(t_train[1:], mu_preds_train, color="b")
    plt.fill_between(
        t_train[1:],
        mu_preds_train - np.sqrt(var_preds_train),
        mu_preds_train + np.sqrt(var_preds_train),
        facecolor="blue",
        alpha=0.3,
    )
    plt.vlines(
        t_train[-1] + pd.Timedelta(hours=t_reset + 1),
        -1.25,
        1.25,
        color="red",
        linestyle="--",
        # label="Reset",
        linewidth=2,
    )
    plt.ylim(-1.25, 1.25)
    plt.legend(loc=(0.01, 1.01), ncol=5, frameon=False)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks([])
    plt.title(plot_title)
    plt.savefig(
        "./lstm_states_example.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    main(
        num_epochs=10,
        sigma_v=0,
        t_reset=-5,  # -1 is the correct value
    )
