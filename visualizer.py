###############################################################################
# File:         visualizer
# Description:  Visualization tool for images data
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      May 10, 2022
# Updated:      May 10, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
###############################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt


class Visualizer:
    """ Visualization of image data"""

    def __init__(self, task_name: str, data_name: str, mu: npt.NDArray,
                 sigma: npt.NDArray, img_size: npt.NDArray) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.mu = mu
        self.sigma = sigma
        self.img_size = img_size

    def load_generated_images(self) -> npt.NDArray:
        """ Load the images generated using TAGI-neural netwrok"""

        # Get file name
        path_dir = f'./saved_results/'
        file_name = f'{path_dir}/generated_images_test.csv'

        # Load image data from *.csv file
        df = pd.read_csv(file_name, skiprows=0, delimiter=",", header=None)
        imgs = df[0].values

        # Reshape data for plot
        num_imgs = int(len(imgs) / np.prod(self.img_size))
        imgs = np.reshape(
            imgs,
            (num_imgs, self.img_size[0], self.img_size[1], self.img_size[2]))
        mu = np.reshape(self.mu, (self.img_size[0], 1, 1))
        sigma = np.reshape(self.sigma, (self.img_size[0], 1, 1))

        imgs = (imgs * self.sigma[0] + self.mu[0]) * 255.0
        imgs = imgs.transpose(0, 2, 3, 1)

        return imgs

    def plot_images(self, n_row: int, n_col: int) -> None:
        """Plot and save figure"""

        # Load images
        imgs = self.load_generated_images()
        (num, _, _, _) = imgs.shape

        # Plot images
        path_dir = './saved_results'
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        fig_path = f'{path_dir}/{self.data_name}_{self.task_name}.png'

        _, axes = plt.subplots(n_row, n_col, figsize=((5, 5)))
        for i in range(num):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(imgs[i], cmap='gray')
            ax.set_axis_off()
        plt.tight_layout()
        #plt.show()
        plt.savefig(fig_path, bbox_inches='tight')


def main():
    # User input data
    task_name = 'autoencoder'
    data_name = 'mnist'
    mu = np.array([0.1309])
    sigma = np.array([1])
    img_size = np.array([1, 28, 28])
    n_row = 10
    n_col = 10

    viz = Visualizer(task_name=task_name,
                     data_name=data_name,
                     mu=mu,
                     sigma=sigma,
                     img_size=img_size)
    viz.plot_images(n_row=n_row, n_col=n_col)


if __name__ == '__main__':
    main()
