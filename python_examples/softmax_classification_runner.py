from python_examples.classification import Classifier
from python_examples.data_loader import MnistOneHotDataloader
from python_examples.model import SoftmaxMnistMLP


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 50
    x_train_file = "./data/mnist/train-images-idx3-ubyte"
    y_train_file = "./data/mnist/train-labels-idx1-ubyte"
    x_test_file = "./data/mnist/t10k-images-idx3-ubyte"
    y_test_file = "./data/mnist/t10k-labels-idx1-ubyte"

    # Model
    net_prop = SoftmaxMnistMLP()

    # Data loader
    cls_data_loader = MnistOneHotDataloader(batch_size=net_prop.batch_size)
    data_loader = cls_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Train and test
    cls_task = Classifier(num_epochs=num_epochs,
                          data_loader=data_loader,
                          net_prop=net_prop,
                          num_classes=10)
    cls_task.train_one_hot()

if __name__ == "__main__":
    main()
