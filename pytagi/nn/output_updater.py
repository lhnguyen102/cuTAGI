import cutagi
import numpy as np

from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates


class OutputUpdater:
    def __init__(self, model_device: str):
        self._cpp_backend = cutagi.OutputUpdater(model_device)

    def update(
        self,
        output_states: BaseHiddenStates,
        mu_obs: np.ndarray,
        var_obs: np.ndarray,
        delta_states: BaseDeltaStates,
    ):
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
        self._cpp_backend.update_using_indices(
            output_states,
            mu_obs.tolist(),
            var_obs.tolist(),
            selected_idx.tolist(),
            delta_states,
        )

    @property
    def device(self) -> str:
        return self._cpp_backend.device

    @device.setter
    def device(self, value: str):
        self._cpp_backend.device = value
