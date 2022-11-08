from python_examples.classification import Classifier
from python_examples.data_loader import MnistDataloader
from python_examples.model import MnistMLP


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 50
    x_train_file = "./data/mnist/train-images-idx3-ubyte"
    y_train_file = "./data/mnist/train-labels-idx1-ubyte"
    x_test_file = "./data/mnist/t10k-images-idx3-ubyte"
    y_test_file = "./data/mnist/t10k-labels-idx1-ubyte"

    # Model
    net_prop = MnistMLP()

    # Data loader
    reg_data_loader = MnistDataloader(batch_size=net_prop.batch_size)
    data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Train and test
    reg_task = Classifier(num_epochs=num_epochs,
                          data_loader=data_loader,
                          net_prop=net_prop,
                          num_classes=10)
    reg_task.train()
    # reg_task.predict()


if __name__ == "__main__":
    main()
