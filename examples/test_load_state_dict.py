# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    ReLU,
    Sequential,
)

# MODEL_1 = Sequential(
#     Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
#     BatchNorm2d(32),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     BatchNorm2d(32),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     Conv2d(32, 64, 5, bias=False, padding=2),
#     BatchNorm2d(64),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     Linear(64 * 4 * 4, 256),
#     ReLU(),
#     Linear(256, 11),
# )

# MODEL_2 = Sequential(
#     Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
#     BatchNorm2d(32),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     BatchNorm2d(32),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     Conv2d(32, 64, 5, bias=False, padding=2),
#     BatchNorm2d(64),
#     ReLU(),
#     AvgPool2d(3, 2, padding=1, padding_type=2),
#     Linear(64 * 4 * 4, 256),
#     ReLU(),
#     Linear(256, 11),
# )


# def test_load_state_dict():
#     # Initialize the model parameter because it is not initialized in the constructor
#     model_1 = MODEL_1
#     model_2 = MODEL_2
#     model_1.preinit_layer()
#     model_2.preinit_layer()

#     # State dict
#     state_dict = model_1.get_state_dict()
#     model_2.load_state_dict(state_dict)

#     # Check if the model 2 is loaded correctly
#     state_dict_2 = model_2.get_state_dict()

#     # state_dict_2["Conv2d(3,32,32,32,5)_0"]["mu_w"][0]
#     assert state_dict == state_dict_2
#     print("LOAD STATE DICT TEST PASSED.")


def test_parameters():
    # Initialize the model parameter because it is not initialized in the constructor
    model_1 = 1
    # model_2 = MODEL_2
    # model_1.preinit_layer()
    # model_2.preinit_layer()

    # params_1 = model_1.parameters()
    # params_2 = model_2.parameters()

    # # Soft update
    # tau = 0.0
    # for param_1, param_2 in zip(params_1, params_2):
    #     for i in range(len(param_1)):
    #         param_1[i] = param_1[i] * tau + (1 - tau) * param_2[i]

    # ## add a test to compare if params_1 = params_2
    # assert all(
    #     (param_1 == param_2).all() for param_1, param_2 in zip(params_1, params_2)
    # )

    # print("PARAMETERS TEST PASSED.")


def main():
    """Test load state dict and params."""
    # test_load_state_dict()
    test_parameters()


if __name__ == "__main__":
    main()
