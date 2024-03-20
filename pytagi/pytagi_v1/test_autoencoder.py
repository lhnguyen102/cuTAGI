import cProfile
import os
import pstats
from io import StringIO
from typing import Tuple, Union

import fire
import matplotlib.pyplot as plt
import memory_profiler
import numpy as np
import numpy.typing as npt
import pandas as pd
from activation import ReLU
from batch_norm import BatchNorm2d
from conv2d import Conv2d
from convtranspose2d import ConvTranspose2d
from data_loader import MnistDataloader
from linear import Linear
from output_updater import OutputUpdater
from pooling import AvgPool2d
from sequential import Sequential
from tqdm import tqdm

from pytagi import Utils, exponential_scheduler

plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
SIGMA_V = 20

ENCODER = Sequential(
    Conv2d(
        1, 16, 3, bias=False, padding=1, in_width=IMAGE_WIDTH, in_height=IMAGE_HEIGHT
    ),
    BatchNorm2d(16),
    ReLU(),
    AvgPool2d(3, 2, 1, 2),
    Conv2d(16, 32, 3, bias=False, padding=1),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, 1, 2),
    Linear(32 * 7 * 7, 100),
    ReLU(),
    Linear(100, 10),
)

DECODER = Sequential(
    Linear(10, 32 * 7 * 7),
    ReLU(),
    ConvTranspose2d(
        32,
        32,
        3,
        bias=True,
        stride=2,
        padding=1,
        padding_type=2,
        in_width=7,
        in_height=7,
    ),
    ReLU(),
    ConvTranspose2d(32, 16, 3, bias=True, stride=2, padding=1, padding_type=2),
    ReLU(),
    ConvTranspose2d(16, 1, 3, bias=True, padding=1),
)


class ImageViz:
    """Visualization of image data
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as mnist or cifar10
        mu: Mean of each input, e.g., for 3 channels; mu: 0.5, 0.5, 0.5
        sigma: Standard deviation of each input
        img_size: Size of image input e.g. mnist size = [28, 28, 1]
    """

    def __init__(
        self,
        task_name: str,
        data_name: str,
        mu: npt.NDArray,
        sigma: npt.NDArray,
        img_size: npt.NDArray,
    ) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.mu = mu
        self.sigma = sigma
        self.img_size = img_size

    def load_generated_images(self) -> npt.NDArray:
        """Load the images generated using TAGI-neural netwrok"""

        # Get file name
        path_dir = f"./saved_results/"
        file_name = f"{path_dir}/generated_images_test.csv"

        # Load image data from *.csv file
        df = pd.read_csv(file_name, skiprows=0, delimiter=",", header=None)
        imgs = df[0].values

        return imgs

    def plot_images(
        self,
        n_row: int,
        n_col: int,
        imgs: Union[None, np.ndarray] = None,
        save_folder: Union[str, None] = None,
    ) -> None:
        """Plot and save figure
        Args:
            n_row: Number of rows for exported image
            n_col: Number of colums for exported image
        """

        # Load images
        if imgs is None:
            imgs = self.load_generated_images()

        # Reshape data for plot
        num_imgs = int(len(imgs) / np.prod(self.img_size))
        imgs = np.reshape(
            imgs, (num_imgs, self.img_size[0], self.img_size[1], self.img_size[2])
        )
        mu = np.reshape(self.mu, (self.img_size[0], 1, 1))
        sigma = np.reshape(self.sigma, (self.img_size[0], 1, 1))

        imgs = (imgs * sigma + mu) * 255.0
        imgs = imgs.transpose(0, 2, 3, 1)

        # Plot images
        _, axes = plt.subplots(n_row, n_col, figsize=((10, 10)))
        for i in range(num_imgs):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(imgs[i], cmap="gray")
            ax.set_axis_off()
        plt.tight_layout()
        if save_folder is None:
            plt.show()
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            fig_path = f"{save_folder}/{self.data_name}_{self.task_name}.png"
            plt.savefig(fig_path, bbox_inches="tight")


class Autoencoder:
    """Test classifier"""

    utils: Utils = Utils()

    def __init__(
        self,
        num_epochs: int,
        data_loader: dict,
        num_classes: int,
        batch_size: int,
        viz: Union[ImageViz, None] = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.encoder = ENCODER
        self.decoder = DECODER

        self.encoder.set_threads(4)
        self.decoder.set_threads(4)
        # self.encoder.to_device("cuda")
        # self.decoder.to_device("cuda")
        self.viz = viz

    def train(self) -> None:
        """Train the network using TAGI"""

        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.decoder.device)

        # Inputs
        batch_size = self.batch_size
        current_sigma_v = SIGMA_V

        # Data
        input_data, _, _, _ = self.data_loader["train"]

        # Progress bar
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs), desc="Training Progress")

        for epoch in pbar:
            for i in range(num_iter):
                # Decaying observation's variance
                current_sigma_v = exponential_scheduler(
                    curr_v=current_sigma_v,
                    min_v=1,
                    decaying_factor=0.99,
                    curr_iter=epoch,
                )
                var_obs = (
                    np.zeros((batch_size, IMAGE_WIDTH * IMAGE_WIDTH), dtype=np.float32)
                    + current_sigma_v**2
                )

                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]

                # Feed forward
                self.encoder(x_batch.flatten())
                self.decoder(self.encoder.output_z_buffer)

                # Update output layer
                output_updater.update(
                    output_states=self.decoder.output_z_buffer,
                    mu_obs=x_batch.flatten(),
                    var_obs=var_obs.flatten(),
                    delta_states=self.decoder.input_delta_z_buffer,
                )

                # Feed backward
                self.decoder.backward()
                self.decoder.step()

                self.encoder.input_delta_z_buffer.copy_from(
                    self.decoder.output_delta_z_buffer
                )

                self.encoder.backward()
                self.encoder.step()
                if i % 1000 == 0 and i > 0:
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{self.num_epochs} | {i * batch_size + len(x_batch):>5}|{num_data: 1}",
                        refresh=True,
                    )
        pbar.close()
        self.predict()

    def predict(self) -> None:
        """Generate images"""
        # Data
        generated_images = []
        for count, (x_batch, _) in enumerate(self.data_loader["test"]):
            # Feed forward
            self.encoder(x_batch.flatten())
            self.decoder(self.encoder.output_z_buffer)

            mu_a, _ = self.decoder.get_outputs()
            generated_images.append(mu_a)

            # Only first 100 images
            if count * self.batch_size > 100:
                break

        generated_images = np.stack(generated_images).flatten()
        generated_images = generated_images[: IMAGE_WIDTH * IMAGE_HEIGHT * 100]

        # Visualization
        if self.viz is not None:
            n_row = 10
            n_col = 10
            self.viz.plot_images(n_row=n_row, n_col=n_col, imgs=generated_images)

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs. TODO: removing hard-coding
        V_batch = (
            np.zeros((batch_size, IMAGE_WIDTH * IMAGE_WIDTH), dtype=np.float32)
            + SIGMA_V**2
        )
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch


# @memory_profiler.profile
def clsf_runner():
    """Run classification training"""
    # User-input
    num_epochs = 1
    batch_size = 20
    mu = np.array([0.1309])
    sigma = np.array([1])
    img_size = np.array([1, IMAGE_WIDTH, IMAGE_HEIGHT])
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

    # Visualization
    viz = ImageViz(
        task_name="autoencoder",
        data_name="mnist",
        mu=mu,
        sigma=sigma,
        img_size=img_size,
    )

    # Train and test
    reg_task = Autoencoder(
        num_epochs=num_epochs,
        data_loader=data_loader,
        num_classes=10,
        batch_size=batch_size,
        viz=viz,
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
