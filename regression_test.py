from python_src.data_loader import RegressionDataLoader
from python_src.regression import Regression


def main():
    """Training and testing API"""
    # User-input
    num_inputs = 1
    num_outputs = 1
    batch_size = 4
    num_epochs = 10
    x_train_file = "./data/toy_example/x_train_1D.csv"
    y_train_file = "./data/toy_example/y_train_1D.csv"
    x_test_file = "./data/toy_example/x_test_1D.csv"
    y_test_file = "./data/toy_example/y_test_1D.csv"

    # Data loader
    reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                           num_outputs=num_outputs,
                                           batch_size=batch_size)
    data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Train and test
    reg_task = Regression(num_epochs=num_epochs, data_loader=data_loader)
    reg_task.train()
    # reg_task.predict()


if __name__ == "__main__":
    main()