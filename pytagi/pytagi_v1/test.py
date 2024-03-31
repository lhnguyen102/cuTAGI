import cProfile
import pstats
from io import StringIO
from typing import Tuple

import fire
import memory_profiler
import numpy as np
from activation import ReLU
from data_loader import MnistDataloader
from linear import Linear
from conv2d import Conv2d
from pooling import AvgPool2d
from output_updater import OutputUpdater
from sequential import Sequential
from tqdm import tqdm
from batch_norm import BatchNorm2d
from layer_norm import LayerNorm

import pytagi.metric as metric
from pytagi import HierarchicalSoftmax, Utils

FNN_NET = Sequential(
    Linear(784, 100),
    ReLU(),
    Linear(100, 100),
    ReLU(),
    Linear(100, 11),
)

FNN_BATCHNORM_NET = Sequential(
    Linear(784, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 11),
)

FNN_LAYERNORM_NET = Sequential(
    Linear(784, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 11),
)

CNN_NET = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_BATCHNORM_NET = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    ReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_LAYERNORM_NET = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    LayerNorm((16, 27, 27)),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    LayerNorm((32, 9, 9)),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)


class Classifier:
    """Test classifier"""

    hr_softmax: HierarchicalSoftmax
    utils: Utils = Utils()

    def __init__(
        self, num_epochs: int, data_loader: dict, num_classes: int, batch_size: int
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.batch_size = batch_size

        # FNN
        self.network = FNN_NET

        self.network.set_threads(4)
        # self.network.to_device("cuda")

    @property
    def num_classes(self) -> int:
        """Get number of classes"""

        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        """Set number of classes"""
        self._num_classes = value
        self.hr_softmax = self.utils.get_hierarchical_softmax(self._num_classes)

    def train(self) -> None:
        """Train the network using TAGI"""

        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.network.device)

        # Inputs
        batch_size = self.batch_size

        # Outputs
        var_obs, _ = self.init_outputs(batch_size)

        # Data
        input_data, output_data, output_idx, labels = self.data_loader["train"]

        # Progress bar
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs), desc="Training Progress")

        error_rates = []
        avg_error_rate = 0
        val_error_rate = np.nan
        for epoch in pbar:
            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                mu_obs_batch = output_data[idx, :]
                ud_idx_batch = output_idx[idx, :]
                label = labels[idx]

                # Feed forward
                self.network(x_batch.flatten())

                # Update output layer
                output_updater.update_using_indices(
                    output_states=self.network.output_z_buffer,
                    mu_obs=mu_obs_batch.flatten(),
                    var_obs=var_obs.flatten(),
                    selected_idx=ud_idx_batch.flatten(),
                    delta_states=self.network.input_delta_z_buffer,
                )

                # Update hidden states
                self.network.backward()
                self.network.step()

                # Error rate
                ma_pred, Sa_pred = self.network.get_outputs()
                pred, _ = self.utils.get_labels(
                    ma=ma_pred,
                    Sa=Sa_pred,
                    hr_softmax=self.hr_softmax,
                    num_classes=self.num_classes,
                    batch_size=batch_size,
                )

                error_rate = metric.classification_error(prediction=pred, label=label)
                error_rates.append(error_rate)

                if i % 1000 == 0 and i > 0:
                    extracted_error_rate = np.hstack(error_rates)
                    avg_error_rate = np.mean(extracted_error_rate[-100:])
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{self.num_epochs} | {i * batch_size + len(x_batch):>5}|{num_data: 1}| training error: {avg_error_rate * 100:.2f}% | validation error: {val_error_rate * 100:.2f}%",
                        refresh=True,
                    )

            # Validate on test set after each epoch
            val_error_rate = self.predict()
            pbar.set_description(
                f"Epoch {epoch + 1}/{self.num_epochs} | {i * batch_size + len(x_batch):>5}|{num_data: 1}| training error: {avg_error_rate * 100:.2f}% | validation error: {val_error_rate * 100:.2f}%",
                refresh=True,
            )

        pbar.close()

    def predict(self) -> None:
        """Make prediction using TAGI"""
        # Inputs

        preds = []
        labels = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network(x_batch.flatten())
            ma, Sa = self.network.get_outputs()
            pred, _ = self.utils.get_labels(
                ma=ma,
                Sa=Sa,
                hr_softmax=self.hr_softmax,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
            )

            # Store data
            preds.append(pred)
            labels.append(y_batch)

        preds = np.stack(preds).flatten()
        labels = np.stack(labels).flatten()

        # Compute classification error rate
        error_rate = metric.classification_error(prediction=preds, label=labels)

        # print("#############")
        # print(f"Error rate    : {error_rate * 100: 0.2f}%")

        return error_rate

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs. TODO: removing hard-coding
        V_batch = (
            np.zeros((batch_size, self.hr_softmax.num_obs), dtype=np.float32) + 1**2
        )
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch


# @memory_profiler.profile
def clsf_runner():
    """Run classification training"""
    # User-input
    num_epochs = 10
    batch_size = 32
    x_train_file = "../../data/mnist/train-images-idx3-ubyte"
    y_train_file = "../../data/mnist/train-labels-idx1-ubyte"
    x_test_file = "../../data/mnist/t10k-images-idx3-ubyte"
    y_test_file = "../../data/mnist/t10k-labels-idx1-ubyte"

    # Data loader
    reg_data_loader = MnistDataloader(batch_size=batch_size)
    data_loader = reg_data_loader.process_data(
        x_train_file=x_train_file,
        y_train_file=y_train_file,
        x_test_file=x_test_file,
        y_test_file=y_test_file,
    )

    # Train and test
    reg_task = Classifier(
        num_epochs=num_epochs,
        data_loader=data_loader,
        num_classes=10,
        batch_size=batch_size,
    )
    reg_task.train()


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
