from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerBlock,
    Linear,
    ReLU,
    ResNetBlock,
    Sequential,
)


def make_layer_block(in_c: int, out_c: int, stride: int = 1, padding_type: int = 1):
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
        ),
        BatchNorm2d(out_c),
        ReLU(),
        Conv2d(out_c, out_c, 3, bias=False, padding=1),
        BatchNorm2d(out_c),
    )


def resnet18_cifar10() -> Sequential:
    """Resnet18 architecture for cifar10"""
    # 32x32
    initial_layers = [
        Conv2d(3, 64, 3, bias=False, padding=1, in_width=32, in_height=32),
        BatchNorm2d(64),
        ReLU(),
    ]

    resnet_layers = [
        # 32x32
        ResNetBlock(make_layer_block(64, 64)),
        ReLU(),
        ResNetBlock(make_layer_block(64, 64)),
        ReLU(),
        # 16x16
        ResNetBlock(
            make_layer_block(64, 128, 2, 2),
            LayerBlock(Conv2d(64, 128, 2, bias=False, stride=2), BatchNorm2d(128)),
        ),
        ReLU(),
        ResNetBlock(make_layer_block(128, 128)),
        ReLU(),
        # 8x8
        ResNetBlock(
            make_layer_block(128, 256, 2, 2),
            LayerBlock(Conv2d(128, 256, 2, bias=False, stride=2), BatchNorm2d(256)),
        ),
        ReLU(),
        ResNetBlock(make_layer_block(256, 256)),
        ReLU(),
        # 4x4
        ResNetBlock(
            make_layer_block(256, 512, 2, 2),
            LayerBlock(Conv2d(256, 512, 2, bias=False, stride=2), BatchNorm2d(512)),
        ),
        ReLU(),
        ResNetBlock(make_layer_block(512, 512)),
        ReLU(),
    ]

    final_layers = [AvgPool2d(4), Linear(512, 11)]

    return Sequential(*initial_layers, *resnet_layers, *final_layers)
