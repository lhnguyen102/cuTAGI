import os
import sys

# path to binding code
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import unittest
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    ReLU,
    Sequential,
)

# Define models for testing
MODEL_1 = Sequential(
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Conv2d(32, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)

MODEL_2 = Sequential(
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Conv2d(32, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)


class TestModelOperations(unittest.TestCase):

    def setUp(self):
        """Initialize models and pre-initialize layers."""
        self.model_1 = MODEL_1
        self.model_2 = MODEL_2
        self.model_1.preinit_layer()
        self.model_2.preinit_layer()

    def test_load_state_dict(self):
        """test loading state_dict."""
        # Get and load state dict
        state_dict = self.model_1.state_dict()
        self.model_2.load_state_dict(state_dict)

        # Verify state dicts are equal
        state_dict_2 = self.model_2.state_dict()
        self.assertEqual(state_dict, state_dict_2, "State dicts are not identical")

    # def test_parameters_soft_update(self):
    #     """test soft parameter update."""
    #     params_1 = self.model_1.parameters()
    #     params_2 = self.model_2.parameters()

    #     # Perform a soft update with tau = 0.0
    #     tau = 0.0
    #     for param_1, param_2 in zip(params_1, params_2):
    #         for i in range(len(param_1)):
    #             param_1[i] = param_1[i] * tau + (1 - tau) * param_2[i]

    #     # Verify the test is only valid for tau = 0.0
    #     self.assertTrue(
    #         all(
    #             (param_1 == param_2).all()
    #             for param_1, param_2 in zip(params_1, params_2)
    #         ),
    #         "Parameters do not match after soft update with tau=0.0",
    #     )


if __name__ == "__main__":
    unittest.main()
