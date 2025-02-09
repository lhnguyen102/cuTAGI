from pytagi.nn import (
    AvgPool2d,
    Conv2d,
    Linear,
    ReLU,
    MixtureReLU,
    Sequential,
    ReLU,
    MaxPool2d,
)


def create_alexnet(gain_w: float = 1, gain_b: float = 1, nb_outputs: int = 1001):
    alex_net = Sequential(
        # 224x224
        Conv2d(
            3,
            64,
            12,
            stride=4,
            padding=2,
            gain_weight=gain_w,
            gain_bias=gain_b,
            in_width=224,
            in_height=224,
            bias=False,
        ),
        ReLU(),
        # 55x55
        AvgPool2d(3, 2),
        # 27x27
        Conv2d(64, 192, 5, bias=False, padding=2, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        # 27x27
        AvgPool2d(3, 2),
        # 13x13
        Conv2d(
            192, 384, 3, bias=False, padding=1, gain_weight=gain_w, gain_bias=gain_b
        ),
        # 13x13
        ReLU(),
        # 13x13
        Conv2d(
            384, 256, 3, bias=False, padding=1, gain_weight=gain_w, gain_bias=gain_b
        ),
        ReLU(),
        # 13x13
        Conv2d(
            256, 256, 3, bias=False, padding=1, gain_weight=gain_w, gain_bias=gain_b
        ),
        ReLU(),
        # 13x13
        AvgPool2d(3, 2),
        # 6x6
        Linear(256 * 6 * 6, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, nb_outputs, gain_weight=gain_w, gain_bias=gain_b),
    )

    return alex_net
