import os
import time

import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

import pytagi.metric as metric
from examples.data_loader import RegressionDataLoader
from examples.time_series_forecasting import PredictionViz
from pytagi import Normalizer
from pytagi.nn import (
    AGVI,
    Exp,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
    SplitActivation,
)


def predict(batch_size, train_dtl, test_dtl, net, cuda):
    test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=True)
    mu_preds = []
    var_preds = []
    y_test = []
    x_test = []

    for x, y in test_batch_iter:
        # Prediction
        m_pred, v_pred = net(x)

        mu_preds.extend(m_pred[::2])
        var_preds.extend(v_pred[::2] + m_pred[1::2])
        # mu_preds.extend(m_pred)
        # var_preds.extend(v_pred)

        x_test.extend(x)
        y_test.extend(y)

    mu_preds = np.array(mu_preds)
    std_preds = np.array(var_preds) ** 0.5
    y_test = np.array(y_test)
    x_test = np.array(x_test)

    mu_preds = Normalizer.unstandardize(
        mu_preds, train_dtl.y_mean, train_dtl.y_std
    )
    std_preds = Normalizer.unstandardize_std(std_preds, train_dtl.y_std)

    y_test = Normalizer.unstandardize(y_test, train_dtl.y_mean, train_dtl.y_std)

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    return mse**0.5, log_lik


def download_and_split_dataset(data_name: str, n_splits: int):
    """
    Downloads the dataset from ucimlrepo, handles preprocessing,
    and creates the necessary train/test split files.
    """
    base_dir = f"./data/UCI_bench/{data_name}"
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Dictionary to map data_name to ucimlrepo id or name
    # You may need to update this for all datasets you use.
    # Check the ucimlrepo documentation for the correct IDs.
    dataset_map = {
        "Boston_housing": 531,
        "Concrete": 165,
        "Energy": 242,
        "Yacht": 266,
        "Wine": 186,
        "Kin8nm": 273,
        "Naval": 643,
        "Power-plant": 294,
        "Protein": 420,
    }

    if data_name in dataset_map:
        dataset = fetch_ucirepo(id=dataset_map[data_name])
    else:
        # Fallback to name if ID is not found
        dataset = fetch_ucirepo(name=data_name)

    if dataset is None:
        raise ValueError(f"Dataset {data_name} not found in ucimlrepo.")

    # Combine features and targets into a single DataFrame
    X = dataset.data.features
    y = dataset.data.targets
    data = pd.concat([X, y], axis=1).dropna()

    # Save the combined data for easy access
    data.to_csv(
        os.path.join(data_dir, "data.txt"), index=False, header=False, sep=" "
    )

    # Generate and save the train/test splits
    for split in range(n_splits):
        np.random.seed(split)  # Ensure reproducibility for each split
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42
        )

        # Save the indices
        np.savetxt(
            os.path.join(data_dir, f"index_train_{split}.txt"),
            train_data.index.values,
            fmt="%d",
        )
        np.savetxt(
            os.path.join(data_dir, f"index_test_{split}.txt"),
            test_data.index.values,
            fmt="%d",
        )

    print(f"Dataset {data_name} downloaded and splits generated.")
    return data_dir


def run_benchmark(data_name: str, num_epochs, batch_size, n_splits):
    """Run benchmark for each UCI dataset"""
    rmse_list = []
    log_lik_list = []
    times_list = []

    # Check and download dataset if not exists
    data_dir = f"./data/UCI_bench/{data_name}/data"
    if not os.path.exists(data_dir) or not os.path.exists(
        os.path.join(data_dir, "data.txt")
    ):
        try:
            download_and_split_dataset(data_name, n_splits)
        except Exception as e:
            print(f"Failed to download {data_name}: {e}")
            return

    num_nodes = 50
    if data_name == "Protein":
        num_nodes = 100
        n_splits = 5
    if data_name == "Yacht":
        batch_size = 5

    for split in range(n_splits):
        # Read and split the data
        data = np.loadtxt(os.path.join(data_dir, "data.txt"))
        train_index = np.loadtxt(
            os.path.join(data_dir, f"index_train_{split}.txt")
        ).astype(int)
        test_index = np.loadtxt(
            os.path.join(data_dir, f"index_test_{split}.txt")
        ).astype(int)

        train_data = data[train_index]
        test_data = data[test_index]

        # Split train data into training and validation
        train_data, val_data = train_test_split(
            train_data, test_size=0.1, random_state=42
        )

        # Save the training and testing data in x and y files
        x_train_file = f"./data/UCI_bench/{data_name}/x_train.csv"
        y_train_file = f"./data/UCI_bench/{data_name}/y_train.csv"
        x_test_file = f"./data/UCI_bench/{data_name}/x_test.csv"
        y_test_file = f"./data/UCI_bench/{data_name}/y_test.csv"
        x_val_file = f"./data/UCI_bench/{data_name}/x_val.csv"
        y_val_file = f"./data/UCI_bench/{data_name}/y_val.csv"

        np.savetxt(x_train_file, train_data[:, :-1], delimiter=",")
        np.savetxt(y_train_file, train_data[:, -1], delimiter=",")
        np.savetxt(x_test_file, test_data[:, :-1], delimiter=",")
        np.savetxt(y_test_file, test_data[:, -1], delimiter=",")
        np.savetxt(x_val_file, val_data[:, :-1], delimiter=",")
        np.savetxt(y_val_file, val_data[:, -1], delimiter=",")

        train_dtl = RegressionDataLoader(
            x_file=x_train_file, y_file=y_train_file
        )
        test_dtl = RegressionDataLoader(
            x_file=x_test_file,
            y_file=y_test_file,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            y_mean=train_dtl.y_mean,
            y_std=train_dtl.y_std,
        )
        val_dtl = RegressionDataLoader(
            x_file=x_val_file,
            y_file=y_val_file,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            y_mean=train_dtl.y_mean,
            y_std=train_dtl.y_std,
        )

        num_inputs = pd.read_csv(x_train_file, header=None).shape[1]
        cuda = True

        net = Sequential(
            Linear(num_inputs, num_nodes),
            ReLU(),
            Linear(num_nodes, num_nodes),
            ReLU(),
            Linear(num_nodes, 2),
            # AGVI(Exp(), overfit_mu=True),
            SplitActivation(Exp()),
        )

        if cuda:
            net.to_device("cuda")
        else:
            net.set_threads(8)

        out_updater = OutputUpdater(net.device)
        delta = 0.01
        patience = 5
        best_log_lik = -np.inf
        counter = 0

        start_time = time.time()

        for epoch in range(num_epochs):
            batch_iter = train_dtl.create_data_loader(batch_size)

            for x, y in batch_iter:
                m_pred, _ = net(x)
                out_updater.update_heteros(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    # var_obs=np.zeros_like(y),
                    delta_states=net.input_delta_z_buffer,
                )
                net.backward()
                net.step()

            rmse, log_lik = predict(batch_size, train_dtl, val_dtl, net, cuda)

            if (log_lik - best_log_lik) > delta:
                best_log_lik = log_lik
                counter = 0
            else:
                counter += 1
                if counter == patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        times_list.append(time.time() - start_time)

        rmse, log_lik = predict(batch_size, train_dtl, test_dtl, net, cuda)

        rmse_list.append(rmse)
        log_lik_list.append(log_lik)

        print(
            f"Split {split + 1}/{n_splits} - RMSE: {rmse: 0.3f}, Log-likelihood: {log_lik: 0.3f}"
        )

    print("#############")
    print(
        f"RMSE           : {np.mean(rmse_list): 0.3f} +- {np.std(rmse_list): 0.3f}"
    )
    print(
        f"Log-likelihood: {np.mean(log_lik_list): 0.3f} +- {np.std(log_lik_list): 0.3f}"
    )
    print(
        f"Time           : {np.mean(times_list): 0.3f} +- {np.std(times_list): 0.3f}"
    )
    print("#############")


def main():
    """Run benchmark for regression tasks on UCI dataset"""

    data_names = [
        "Boston_housing",
        "Concrete",
        "Energy",
        "Yacht",
        "Wine",
        "Kin8nm",
        "Naval",
        "Power-plant",
        "Protein",
    ]
    # data_names = ["Protein"]

    for data_name in data_names:
        print(f"Running benchmark for {data_name}")
        run_benchmark(data_name, num_epochs=100, batch_size=10, n_splits=20)


if __name__ == "__main__":
    fire.Fire(main)
