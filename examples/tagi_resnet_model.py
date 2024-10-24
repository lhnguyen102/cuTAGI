from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerBlock,
    Linear,
    MixtureReLU,
    ResNetBlock,
    Sequential,
)


def make_layer_block(in_c: int, out_c: int, stride: int = 1, padding_type: int = 1, gain_weight: float = 1, gain_bias: float = 1):
    """Create a layer block for resnet 18"""

    return LayerBlock(
        Conv2d(
            in_c,
            out_c,
            3,
            bias=False,
            stride=stride,
            padding=1,
            padding_type=padding_type,
            gain_weight = gain_weight,
            gain_bias = gain_bias,
        ),
        MixtureReLU(),
        BatchNorm2d(out_c),
        Conv2d(out_c,
               out_c, 3,
               bias=False,
               padding=1,
               gain_weight = gain_weight,
               gain_bias = gain_bias),
        MixtureReLU(),
        BatchNorm2d(out_c),
    )


def resnet18_cifar10(gain_w: float = 1, gain_b: float = 1) -> Sequential:
    """Resnet18 architecture for cifar10"""
    # 32x32
    initial_layers = [
        Conv2d(3, 64, 3,
               bias=False,
               padding=1,
               in_width=32, in_height=32,
               gain_weight = gain_w,
               gain_bias = gain_b),
        MixtureReLU(),
        BatchNorm2d(64),
    ]

    resnet_layers = [
        # 32x32
        ResNetBlock(make_layer_block(64, 64, gain_weight=gain_w, gain_bias=gain_b)),
        ResNetBlock(make_layer_block(64, 64, gain_weight=gain_w, gain_bias=gain_b)),
        # 16x16
        ResNetBlock(
            make_layer_block(64, 128, 2, 2, gain_weight=gain_w, gain_bias=gain_b),
            LayerBlock(Conv2d(64, 128, 2, bias=False, stride=2,
            gain_weight = gain_w,
            gain_bias = gain_b), BatchNorm2d(128))
        ),
        ResNetBlock(make_layer_block(128, 128, gain_weight=gain_w, gain_bias=gain_b)),
        # 8x8
        ResNetBlock(
            make_layer_block(128, 256, 2, 2, gain_weight=gain_w, gain_bias=gain_b),
            LayerBlock(Conv2d(128, 256, 2, bias=False, stride=2,
            gain_weight = gain_w,
            gain_bias = gain_b), BatchNorm2d(256))
        ),
        ResNetBlock(make_layer_block(256, 256, gain_weight=gain_w, gain_bias=gain_b)),
        # 4x4
        ResNetBlock(
            make_layer_block(256, 512, 2, 2, gain_weight=gain_w, gain_bias=gain_b),
            LayerBlock(Conv2d(256, 512, 2, bias=False, stride=2,
            gain_weight = gain_w,
            gain_bias = gain_b), BatchNorm2d(512))
        ),
        ResNetBlock(make_layer_block(512, 512, gain_weight=gain_w, gain_bias=gain_b))
    ]

    final_layers = [AvgPool2d(4),
                    Linear(512, 11, gain_weight = gain_b, gain_bias = gain_b)
    ]

    return Sequential(*initial_layers, *resnet_layers, *final_layers)
