import cutagi
import numpy as np

from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates


class OutputUpdater:
    """A utility to compute the error signal (delta states) for the output layer.

    This class calculates the difference between the model's predictions and the
    observations, which is essential for performing the backward pass
    to update the model's parameters. It wraps the C++/CUDA backend `cutagi.OutputUpdater`.
    """

    def __init__(self, model_device: str):
        """Initializes the OutputUpdater.

        :param model_device: The computational device the model is on (e.g., 'cpu' or 'cuda:0').
        :type model_device: str
        """
        self._cpp_backend = cutagi.OutputUpdater(model_device)

    def update(
        self,
        output_states: BaseHiddenStates,
        mu_obs: np.ndarray,
        var_obs: np.ndarray,
        delta_states: BaseDeltaStates,
    ):
        """Computes the delta states based on observations.

        This method is used for homoscedastic regression where the observation
        variance is known and provided.

        :param output_states: The hidden states (mean and variance) of the model's output layer.
        :type output_states: pytagi.nn.data_struct.BaseHiddenStates
        :param mu_obs: The mean of the ground truth observations.
        :type mu_obs: np.ndarray
        :param var_obs: The variance of the ground truth observations.
        :type var_obs: np.ndarray
        :param delta_states: The delta states object to be updated with the computed error signal.
        :type delta_states: pytagi.nn.data_struct.BaseDeltaStates
        """
        self._cpp_backend.update(
            output_states, mu_obs.tolist(), var_obs.tolist(), delta_states
        )

    def update_using_indices(
        self,
        output_states: BaseHiddenStates,
        mu_obs: np.ndarray,
        var_obs: np.ndarray,
        selected_idx: np.ndarray,
        delta_states: BaseDeltaStates,
    ):
        """Computes the delta states for a selected subset of outputs.

        This is useful in scenarios like hierarchical softmax or when only
        a sparse set of outputs needs to be updated.

        :param output_states: The hidden states of the model's output layer.
        :type output_states: pytagi.nn.data_struct.BaseHiddenStates
        :param mu_obs: The mean of the ground truth observations.
        :type mu_obs: np.ndarray
        :param var_obs: The variance of the ground truth observations.
        :type var_obs: np.ndarray
        :param selected_idx: An array of indices specifying which output neurons to update.
        :type selected_idx: np.ndarray
        :param delta_states: The delta states object to be updated with the computed error signal.
        :type delta_states: pytagi.nn.data_struct.BaseDeltaStates
        """
        self._cpp_backend.update_using_indices(
            output_states,
            mu_obs.tolist(),
            var_obs.tolist(),
            selected_idx.tolist(),
            delta_states,
        )

    def update_heteros(
        self,
        output_states: BaseHiddenStates,
        mu_obs: np.ndarray,
        delta_states: BaseDeltaStates,
    ):
        """Computes delta states for heteroscedastic regression.

        In this case, the model is expected to predict both the mean and the variance
        of the output. The predicted variance is taken from the `output_states`.

        :param output_states: The hidden states of the model's output layer. The model's
                              predicted variance is sourced from here.
        :type output_states: pytagi.nn.data_struct.BaseHiddenStates
        :param mu_obs: The mean of the ground truth observations.
        :type mu_obs: np.ndarray
        :param delta_states: The delta states object to be updated with the computed error signal.
        :type delta_states: pytagi.nn.data_struct.BaseDeltaStates
        """
        self._cpp_backend.update_heteros(
            output_states, mu_obs.tolist(), delta_states
        )

    @property
    def device(self) -> str:
        """The computational device ('cpu' or 'cuda') the updater is on."""
        return self._cpp_backend.device

    @device.setter
    def device(self, value: str):
        """Sets the computational device.

        :param value: The device to set, e.g., 'cpu' or 'cuda:0'.
        :type value: str
        """
        self._cpp_backend.device = value
