import fire
import numpy as np
from tqdm import tqdm

import pytagi
import pytagi.metric as metric
from examples.data_loader import RegressionDataLoader
from examples.time_series_forecasting import PredictionViz
from pytagi import Normalizer
from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential
from matplotlib import pyplot as plt

def main(num_epochs: int = 40, batch_size: int = 10, sigma_v: float = 0.01):
    """Run training for the regression"""
    # Dataset
    x_train_file = "./data/toy_example/x_train_1D.csv"
    y_train_file = "./data/toy_example/y_train_1D.csv"
    x_test_file = "./data/toy_example/x_test_1D.csv"
    y_test_file = "./data/toy_example/y_test_1D.csv"

    train_dtl = RegressionDataLoader(x_file=x_train_file, y_file=y_train_file)
    test_dtl = RegressionDataLoader(
        x_file=x_test_file,
        y_file=y_test_file,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        y_mean=train_dtl.y_mean,
        y_std=train_dtl.y_std,
    )
    seed = 48
    np.random.seed(seed=seed)
    # Add noise to training data
    x_train, y_train = train_dtl.dataset["value"]
    y_train += np.random.normal(0, sigma_v, size=y_train.shape)
    train_dtl.dataset["value"] = (x_train, y_train)

    x_test, y_test = test_dtl.dataset["value"]
    y_test += np.random.normal(0, sigma_v, size=y_test.shape)
    test_dtl.dataset["value"] = (x_test, y_test)
    # Viz
    viz = PredictionViz(task_name="regression", data_name="1d_toy")

    pytagi.manual_seed(seed)


    # Network
    net = Sequential(
        Linear(1, 50),
        ReLU(),
        Linear(50, 1),
    )
    # net.to_device("cuda")
    # net.set_threads(8)

    out_updater = OutputUpdater(net.device)
    var_y = np.full((batch_size,), sigma_v**2, dtype=np.float32)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    mses2 = []
    Pr_es = []
    Pr_es_test = []
    lls = []
    lls_test = []
    se_test = []
    nb_obs = train_dtl.dataset["value"][1].size
    marg_lik_s = np.zeros(shape = (nb_obs,num_epochs))
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)
        marg_lik = []
        ll = 0.0
        #ll_test = 0.0
        #se_test = 0.0
        se = 0.0
        for x, y in batch_iter:
            # Training
            # Feed forward
            m_pred, v_pred = net(x)

            # Update output layer
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Feed backward
            net.backward()
            net.step()

            # Training metric
            pred = Normalizer.unstandardize(
                m_pred, train_dtl.y_mean, train_dtl.y_std
            )
            obs = Normalizer.unstandardize(y, train_dtl.y_mean, train_dtl.y_std)
            mse = metric.mse(pred, obs)
            mses.append(mse)
            se += metric.mse(pred, obs)
            ll += metric.log_likelihood(pred, obs, np.sqrt(v_pred + sigma_v**2))
            marg_lik_b = metric.marg_likelihood(pred, obs, np.sqrt(v_pred+sigma_v**2))
            if len(marg_lik) == 0:
                marg_lik = marg_lik_b.flatten()
            else:
                marg_lik = np.concatenate((marg_lik, marg_lik_b.flatten()))

        marg_lik_s[:, epoch] = marg_lik.flatten()
        lls.append(ll)
        mses2.append(se)

        # Test
        test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []
        for x, y in test_batch_iter:
            # Predicion
            m_pred, v_pred = net(x)
            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)
            x_test.extend(x)
            y_test.extend(y)
        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # Compute log-likelihood
        se = metric.se(mu_preds, y_test)
        se_test.append(se)
        lls_test.append(metric.log_likelihood(mu_preds, y_test, std_preds))

        # Visualization
        if epoch in [1, 2, 4, 8, 16, 32, 64, 128]:
            plot = True
        else:
            plot = False
        if plot:
            viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=x_test,
                y_test=y_test,
                y_pred=mu_preds,
                sy_pred=std_preds,
                std_factor=3,
                label="diag",
                title=["Epoch #",epoch],
            )

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.2f}",
            refresh=True,
        )

    Pr_sum = np.sum(marg_lik_s, axis=1)
    a = np.sum(np.maximum(marg_lik_s,1E330) / Pr_sum[:,None], axis=0)
    Pr_es = a / nb_obs

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(151)
    ax1.plot(Pr_es, label='EBMS')
    ls = np.exp(lls-np.max(lls))
    ax1.plot(ls/np.sum(ls), label='BMS')
    ls_test = np.exp(lls_test-np.max(lls_test))
    ax1.plot(ls_test /np.sum(ls_test), label='BMS - Test')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Pr(e|D)')
    ax1.legend()

    ax2 = fig.add_subplot(152)
    ax2.plot(mses2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('SE')

    ax3 = fig.add_subplot(153)
    ax3.plot(se_test)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('SE')

    ax4 = fig.add_subplot(154)
    ax4.plot(lls_test)
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Log-likelihood')

    ax5 = fig.add_subplot(155)
    ax5.plot(lls-np.max(lls), label='BMS')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('Log-likelihood')

    plt.show()

    # -------------------------------------------------------------------------#
    # Testing
    test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
    mu_preds = []
    var_preds = []
    y_test = []
    x_test = []

    for x, y in test_batch_iter:
        # Predicion
        m_pred, v_pred = net(x)

        mu_preds.extend(m_pred)
        var_preds.extend(v_pred + sigma_v**2)
        x_test.extend(x)
        y_test.extend(y)

    mu_preds = np.array(mu_preds)
    std_preds = np.array(var_preds) ** 0.5
    y_test = np.array(y_test)
    x_test = np.array(x_test)

    mu_preds = Normalizer.unstandardize(
        mu_preds, train_dtl.y_mean, train_dtl.y_std
    )
    std_preds = Normalizer.unstandardize_std(std_preds, train_dtl.y_std)

    x_test = Normalizer.unstandardize(x_test, train_dtl.x_mean, train_dtl.x_std)
    y_test = Normalizer.unstandardize(y_test, train_dtl.y_mean, train_dtl.y_std)

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    # Visualization
    viz.plot_predictions(
        x_train=None,
        y_train=None,
        x_test=x_test,
        y_test=y_test,
        y_pred=mu_preds,
        sy_pred=std_preds,
        std_factor=3,
        label="diag",
        title="Diagonal covariance",
    )

    print("#############")
    print(f"MSE           : {mse: 0.2f}")
    print(f"Log-likelihood: {log_lik: 0.2f}")


if __name__ == "__main__":
    fire.Fire(main)
