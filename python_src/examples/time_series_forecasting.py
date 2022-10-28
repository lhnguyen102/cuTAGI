from python_src.data_loader import TimeSeriesDataloader
from python_src.model import TimeSeriesLSTM
from python_src.time_series_forecaster import TimeSeriesForecaster
from visualizer import PredictionViz


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 20
    output_col = [0]
    num_features = 1
    input_seq_len = 5
    output_seq_len = 1
    seq_stride = 1
    x_train_file = "./data/toy_time_series/x_train_sin_data.csv"
    datetime_train_file = "./data/toy_time_series/train_sin_datetime.csv"
    x_test_file = "./data/toy_time_series/x_test_sin_data.csv"
    datetime_test_file = "./data/toy_time_series/test_sin_datetime.csv"

    # Model
    net_prop = TimeSeriesLSTM(input_seq_len=input_seq_len,
                              output_seq_len=output_seq_len,
                              seq_stride=seq_stride)

    # Data loader
    ts_data_loader = TimeSeriesDataloader(batch_size=net_prop.batch_size,
                                          output_col=output_col,
                                          input_seq_len=input_seq_len,
                                          output_seq_len=output_seq_len,
                                          num_features=num_features,
                                          stride=seq_stride)
    data_loader = ts_data_loader.process_data(
        x_train_file=x_train_file,
        datetime_train_file=datetime_train_file,
        x_test_file=x_test_file,
        datetime_test_file=datetime_test_file)

    # Visualzier
    viz = PredictionViz(task_name="forecasting", data_name="sin_signal")

    # Train and test
    reg_task = TimeSeriesForecaster(num_epochs=num_epochs,
                                    data_loader=data_loader,
                                    net_prop=net_prop,
                                    viz=viz)
    reg_task.train()
    reg_task.predict()


if __name__ == "__main__":
    main()
