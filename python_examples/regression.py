###############################################################################
# File:         regression.py
# Description:  Example of regression task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      November 07, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from typing import Union, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


import pytagi.metric as metric
from pytagi import NetProp, TagiNetwork
from pytagi import Normalizer as normalizer
from pytagi import Utils, exponential_scheduler
from visualizer import PredictionViz


class Regression:
    """Regression task using TAGI"""

    utils: Utils = Utils()

    def __init__(
        self,
        num_epochs: int,
        data_loader: dict,
        net_prop: NetProp,
        dtype=np.float32,
        viz: Union[PredictionViz, None] = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
        self.network = TagiNetwork(self.net_prop)
        self.dtype = dtype
        self.viz = viz

    def train(self) -> None:
        """Train the network using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, ud_idx_batch = self.init_outputs(batch_size)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            # Decaying observation's variance
            self.net_prop.sigma_v = exponential_scheduler(
                curr_v=self.net_prop.sigma_v,
                min_v=self.net_prop.sigma_v_min,
                decaying_factor=self.net_prop.decay_factor_sigma_v,
                curr_iter=epoch,
            )
            V_batch = V_batch * 0.0 + self.net_prop.sigma_v**2

            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch, ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Loss
                norm_pred, _ = self.network.get_network_predictions()
                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                mse = metric.mse(pred, obs)
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )

    def predict(self, std_factor: int = 1) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_predictions()

            mean_predictions.append(ma)
            variance_predictions.append(Sa + self.net_prop.sigma_v**2)
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten()) ** 0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        # Unnormalization
        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )
        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"]
        )

        x_test = normalizer.unstandardize(
            norm_data=x_test,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"],
        )
        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        # Visualization
        if self.viz is not None:
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=x_test,
                y_test=y_test,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=std_factor,
                label="diag",
                title="Diagonal covariance",
            )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")

    def train_UCI(self) -> None:
        """Train the network using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, ud_idx_batch = self.init_outputs(batch_size)

        input_data, output_data = self.data_loader["train"]


        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))

        # mse, rmse, ll epochlist initialization
        mse_Epochlist, rmse_Epochlist, LL_Epochlist, normal_LL_Epochlist = [], [], [], []
        # initialize param for early-stopping
        if self.net_prop.early_stop == 1:
            delta = self.net_prop.delta
            patience = self.net_prop.patience
            best_LL_val  = np.nan
            counter = 0
            maxEpoch = self.num_epochs
        else:
            stop_epoch = self.num_epochs


        for epoch in pbar:

            mse_list, rmse_list, LL_list = [], [], []
            for i in range(num_iter+1):
                # Get data
                if i<= num_iter-1:
                    # start_idx = i*batch_size
                    # end_idx = min((i + 1) * batch_size, num_iter*batch_size)
                    # x_batch = input_data[start_idx:end_idx, :]
                    # y_batch = output_data[start_idx:end_idx, :]
                    idx = np.random.choice(num_data, size=batch_size)
                    x_batch = input_data[idx, :]
                    y_batch = output_data[idx, :]
                else:
                    start_idx = num_iter*batch_size
                    remaining_len = num_data - num_iter*batch_size
                    fill_idx = np.random.choice(num_data-remaining_len,batch_size-remaining_len)
                    idx = list(range(start_idx, num_data))
                    idx.extend(fill_idx)
                    x_batch = input_data[idx, :]
                    y_batch = output_data[idx, :]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # outputs from the network
                ma, va = self.network.get_network_outputs()

                # print(f"The expected values are: {ma}")
                # print(f"The variance values are: {va}")


                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch, ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Loss
                norm_pred, v_pred = self.network.get_network_predictions()

                # print(f"The normalized predictions are: {norm_pred}")
                # print(f"The variance predictions are: {v_pred}")

                # normalized log-likelihood
                # normal_log_lik = metric.log_likelihood(
                #     prediction=pred, observation=obs, std=np.sqrt(v_pred)
                # )

                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                std_pred = normalizer.unstandardize_std(
                    norm_std=np.sqrt(v_pred), std=self.data_loader["y_norm_param_2"]
                )
                # print(std_pred**2)
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                mse = metric.mse(pred, obs)
                rmse = mse**0.5
                log_lik = metric.log_likelihood(
                    prediction=pred, observation=obs, std=std_pred
                )
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t rmse: {rmse:>7.2f}"
                )
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t log_lik: {log_lik:>7.2f}"
                )




                # save the values
                # mse_list += [mse]
                # rmse_list += [rmse]
                # LL_list += [log_lik]

            # using val data for early stopping
            if self.net_prop.early_stop == 1:
                _, LL_val, _, _ = self.predict_val_UCI()
                if np.isnan(LL_val).any():
                    LL_val[np.isnan(LL_val)] = -np.inf


            ## Early-Stop
            if self.net_prop.early_stop == 1:
                if not np.isnan(LL_val):
                    if np.isnan(best_LL_val):
                        best_LL_val = LL_val
                    elif (LL_val - best_LL_val) > delta:
                        best_LL_val = LL_val
                        counter = 0
                    elif (LL_val - best_LL_val) < delta:
                        counter += 1
                        print(f'   counter #{counter} out of {patience}')
                        if counter >= patience:
                            maxEpoch = epoch


                else:
                    break



            # testing the trained model after each epoch
            mse_Epoch, LL_Epoch, rmse_Epoch, normal_LL_Epoch = self.predict_UCI()

            mse_Epochlist +=[mse_Epoch]
            rmse_Epochlist += [rmse_Epoch]
            LL_Epochlist += [LL_Epoch]
            normal_LL_Epochlist += [normal_LL_Epoch]

            if self.net_prop.early_stop == 1:
                if epoch >= maxEpoch-1:
                    stop_epoch = maxEpoch
                    print(f"The early stopping epoch is {stop_epoch}")
                    break

            # # saving the mean values for the metric for each epoch
            # mse_Epoch = np.mean(mse_list)
            # rmse_Epoch = np.mean(rmse_list)
            # LL_Epoch = np.mean(LL_list)

        return mse_Epochlist, rmse_Epochlist, LL_Epochlist, normal_LL_Epochlist, stop_epoch

    def predict_UCI(self, std_factor: int = 1) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_predictions()
            print(f"The mean predictions are: {ma}")
            print(f"The variance predictions are: {Sa}")

            mean_predictions.append(ma)
            variance_predictions.append(Sa + self.net_prop.sigma_v**2)
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten()) ** 0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        # normalised log-likelihood
        normal_log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        # Unnormalization
        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )
        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"]
        )

        # x_test = normalizer.unstandardize(
        #     norm_data=x_test,
        #     mu=self.data_loader["x_norm_param_1"],
        #     std=self.data_loader["x_norm_param_2"],
        # )
        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        rmse = mse**0.5
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        # Visualization
        if self.viz is not None:
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=x_test,
                y_test=y_test,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=std_factor,
                label="diag",
                title="Diagonal covariance",
            )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")
        print(f"RMSE          : {rmse: 0.2f}")

        return mse, log_lik, rmse, normal_log_lik

    def predict_val_UCI(self, std_factor: int = 1) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_predictions = []
        variance_predictions = []
        y_val = []
        x_val = []
        for x_batch, y_batch in self.data_loader["val"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_predictions()
            # print(f"The mean predictions are: {ma}")
            # print(f"The variance predictions are: {Sa}")

            mean_predictions.append(ma)
            variance_predictions.append(Sa + self.net_prop.sigma_v**2)
            x_val.append(x_batch)
            y_val.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten()) ** 0.5
        y_val = np.stack(y_val).flatten()
        x_val = np.stack(x_val).flatten()

        # normalised log-likelihood
        normal_log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_val, std=std_predictions
        )

        # Unnormalization
        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )
        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"]
        )

        # x_test = normalizer.unstandardize(
        #     norm_data=x_test,
        #     mu=self.data_loader["x_norm_param_1"],
        #     std=self.data_loader["x_norm_param_2"],
        # )
        y_val = normalizer.unstandardize(
            norm_data=y_val,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_val)
        rmse = mse**0.5
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_val, std=std_predictions
        )

        # Visualization
        if self.viz is not None:
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_val=x_val,
                y_val=y_val,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=std_factor,
                label="diag",
                title="Diagonal covariance",
            )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")
        print(f"RMSE          : {rmse: 0.2f}")

        return mse, log_lik, rmse, normal_log_lik


    def compute_derivatives(
        self, layer: int = 0, truth_derv_file: Union[None, str] = None
    ) -> None:
        """Compute dervative of a given layer"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_derv = []
        variance_derv = []
        x_test = []
        for x_batch, _ in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            mdy, vdy = self.network.get_derivatives(layer)

            mean_derv.append(mdy)
            variance_derv.append(vdy)
            x_test.append(x_batch)

        mean_derv = np.stack(mean_derv).flatten()
        std_derv = (np.stack(variance_derv).flatten()) ** 0.5
        x_test = np.stack(x_test).flatten()

        # Unnormalization
        x_test = normalizer.unstandardize(
            norm_data=x_test,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"],
        )

        if truth_derv_file is not None:
            truth_dev_test = pd.read_csv(
                truth_derv_file, skiprows=1, delimiter=",", header=None
            )
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=x_test,
                y_test=truth_dev_test.values,
                y_pred=mean_derv,
                sy_pred=std_derv,
                std_factor=3,
                label="deriv",
                title="Neural Network's Derivative",
            )

    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, self.net_prop.nodes[0]), dtype=self.dtype)

        Sx_f_batch = np.array([], dtype=self.dtype)
        if self.net_prop.is_full_cov:
            Sx_f_batch = self.utils.get_upper_triu_cov(
                batch_size=batch_size,
                num_data=self.net_prop.nodes[0],
                sigma=self.net_prop.sigma_x,
            )
            Sx_batch = Sx_batch + self.net_prop.sigma_x**2

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = (
            np.zeros((batch_size, self.net_prop.nodes[-1]), dtype=self.dtype)
            + self.net_prop.sigma_v**2
        )
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch
