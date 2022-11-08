import numpy as np
from visualizer import ImageViz

from python_examples.autoencoder import Autoencoder
from python_examples.data_loader import MnistDataloader
from python_examples.model import MnistDecoder, MnistEncoder


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 10
    mu = np.array([0.1309])
    sigma = np.array([1])
    img_size = np.array([1, 28, 28])
    x_train_file = "./data/mnist/train-images-idx3-ubyte"
    y_train_file = "./data/mnist/train-labels-idx1-ubyte"
    x_test_file = "./data/mnist/t10k-images-idx3-ubyte"
    y_test_file = "./data/mnist/t10k-labels-idx1-ubyte"

    # Model
    encoder_prop = MnistEncoder()
    decoder_prop = MnistDecoder()

    # Data loader
    reg_data_loader = MnistDataloader(batch_size=encoder_prop.batch_size)
    data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Visualization
    viz = ImageViz(task_name="autoencoder",
                   data_name="mnist",
                   mu=mu,
                   sigma=sigma,
                   img_size=img_size)

    # Train and test
    reg_task = Autoencoder(num_epochs=num_epochs,
                           data_loader=data_loader,
                           encoder_prop=encoder_prop,
                           decoder_prop=decoder_prop,
                           viz=viz)
    reg_task.train()
    reg_task.predict()


if __name__ == "__main__":
    main()
