import numpy as np
from pytagi.nn import Linear, OutputUpdater, Sequential


# Define function to calculate the posterior
def calc_posterior(x, var_x, y, var_y):
    return x + (var_x / (var_x + var_y)) * (y - x), (1 - var_x / (var_x + var_y)) * var_x


def main():
    # Create synthetic data
    y = np.array([1], dtype=np.float32)  # observation
    x = np.array([0.5], dtype=np.float32)  # prior mean
    var_x = np.array([0.01], dtype=np.float32)  # prior variance
    var_y = np.array([0.01], dtype=np.float32)  # observation variance

    # Calculate the posterior using the analytical solution
    x_post, var_post = calc_posterior(x, var_x, y, var_y)

    # Calculate the posterior using the network
    net = Sequential(
        Linear(1, 1),
    )
    # net.to_device("cuda")
    net.input_state_update = True  # enables the input state update
    out_updater = OutputUpdater(net.device)

    # Initialize the network with w=1, b=0, var_w=0, var_b=0
    state_dict = net.get_state_dict()
    for key in state_dict.keys():
        state_dict[key]["mu_w"] = np.ones_like(state_dict[key]["mu_w"])  # reset the weights
        state_dict[key]["mu_b"] = np.zeros_like(state_dict[key]["mu_b"])  # reset the bias
        state_dict[key]["var_w"] = np.zeros_like(state_dict[key]["var_w"])  # variance of the weights = 0
        state_dict[key]["var_b"] = np.zeros_like(state_dict[key]["var_b"])  # variance of the bias = 0
    net.load_state_dict(state_dict)

    # Forward pass
    m_pred, _ = net.forward(x, var_x)

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

    # Get the input states and update the states
    (delta_x, delta_var) = net.get_input_states()
    x = x + delta_x * var_x  # update the mean
    var_x = var_x + var_x * delta_var * var_x  # update the variance

    # Check if the updates match the analytical solution
    if np.allclose(x_post, x) and np.allclose(var_post, var_x):
        print("The updates match the analytical solution!")
    else:
        print("The updates do not match the analytical solution.")


if __name__ == "__main__":
    main()
