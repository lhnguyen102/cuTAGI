from visualizer import PredictionViz

from python_examples.data_loader import TimeSeriesDataloader
from python_examples.model import TimeSeriesLSTM
from python_examples.time_series_forecaster import TimeSeriesForecaster
from pytagi  import load_param_from_files


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 50
    output_col = [0]
    num_features = 1
    input_seq_len = 5
    output_seq_len = 1
    seq_stride = 1
    x_train_file = "./data/toy_time_series/x_train_sin_data.csv"
    datetime_train_file = "./data/toy_time_series/train_sin_datetime.csv"
    x_test_file = "./data/toy_time_series/x_test_sin_data.csv"
    datetime_test_file = "./data/toy_time_series/test_sin_datetime.csv"

    # Load pretrained weights and biases
    mw_file = "./saved_param/lstm_demo_2lstm_1_mw.csv"
    Sw_file = "./saved_param/lstm_demo_2lstm_2_Sw.csv"
    mb_file = "./saved_param/lstm_demo_2lstm_3_mb.csv"
    Sb_file = "./saved_param/lstm_demo_2lstm_4_Sb.csv"
    mw_sc_file = "./saved_param/lstm_demo_2lstm_5_mw_sc.csv"
    Sw_sc_file = "./saved_param/lstm_demo_2lstm_6_Sw_sc.csv"
    mb_sc_file = "./saved_param/lstm_demo_2lstm_7_mb_sc.csv"
    Sb_sc_file = "./saved_param/lstm_demo_2lstm_8_Sb_sc.csv"
    param = load_param_from_files(mw_file=mw_file,
                                  Sw_file=Sw_file,
                                  mb_file=mb_file,
                                  Sb_file=Sb_file,
                                  mw_sc_file=mw_sc_file,
                                  Sw_sc_file=Sw_sc_file,
                                  mb_sc_file=mb_sc_file,
                                  Sb_sc_file=Sb_sc_file)

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
                                    param=param,
                                    viz=viz)
    reg_task.train()
    reg_task.predict()


if __name__ == "__main__":
    main()
