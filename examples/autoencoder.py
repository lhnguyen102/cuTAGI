import os
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from examples.data_loader import MnistDataLoader
from pytagi import exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)


plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


def main(num_epochs: int = 2, batch_size: int = 20, sigma_v: float = 16.0):
    """Run autoencoder training"""
    # Initalization
    mu = np.array([0.1309])
    sigma = np.array([1])
    img_size = np.array([1, IMAGE_WIDTH, IMAGE_HEIGHT])
    num_pixels = IMAGE_HEIGHT * IMAGE_WIDTH

    # Load dataset
    num_train_images = 60000
    train_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=num_train_images,
    )
    test_dtl = MnistDataLoader(
        x_file="data/mnist/t10k-images-idx3-ubyte",
        y_file="data/mnist/t10k-labels-idx1-ubyte",
        num_images=10000,
    )

    # Visualization
    viz = ImageViz(
        task_name="autoencoder",
        data_name="mnist",
        mu=mu,
        sigma=sigma,
        img_size=img_size,
    )

    # Network
    encoder = Sequential(
        Conv2d(
            1,
            16,
            3,
            bias=False,
            padding=1,
            in_width=IMAGE_WIDTH,
            in_height=IMAGE_HEIGHT,
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

    decoder = Sequential(
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
    # encoder.to_device("cuda")
    # decoder.to_device("cuda")
    # encoder.set_threads(8)
    # decoder.set_threads(8)

    out_updater = OutputUpdater(decoder.device)

    var_y = np.full((batch_size * num_pixels,), sigma_v**2, dtype=np.float32)

    # -------------------------------------------------------------------------#
    # Training
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * num_pixels,), sigma_v**2, dtype=np.float32)

        for i, (x, _, _, _) in enumerate(batch_iter):
            # Feed forward
            _, _ = encoder(x)
            _, _ = decoder(encoder.output_z_buffer)

            # Update output layer
            out_updater.update(
                output_states=decoder.output_z_buffer,
                mu_obs=x,
                var_obs=var_y,
                delta_states=decoder.input_delta_z_buffer,
            )

            # Feed backward
            decoder.backward()
            decoder.step()

            # Send updating values to encoder
            encoder.input_delta_z_buffer.copy_from(decoder.output_delta_z_buffer)

            encoder.backward()
            encoder.step()
            if i % 1000 == 0 and i > 0:
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | {i * batch_size + len(x):>5}|{num_train_images: 1}",
                    refresh=True,
                )
    pbar.close()

    # -------------------------------------------------------------------------#
    # Generate images
    test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
    generated_images = []
    for count, (x_test, _, _, _) in enumerate(test_batch_iter):
        # Feed forward
        _, _ = encoder(x_test)
        m_pred, _ = decoder(encoder.output_z_buffer)

        generated_images.append(m_pred)

        # Only first 100 images
        if count * batch_size > 100:
            break

    generated_images = np.stack(generated_images).flatten()
    generated_images = generated_images[: num_pixels * 100]

    # Visualization
    n_row = 10
    n_col = 10
    viz.plot_images(n_row=n_row, n_col=n_col, imgs=generated_images)


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
        imgs: Optional[np.ndarray] = None,
        save_folder: Optional[str] = None,
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


if __name__ == "__main__":
    fire.Fire(main)
