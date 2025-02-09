from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerBlock,
    Linear,
    MixtureReLU,
    ResNetBlock,
    Sequential,
    ReLU,
)
import numpy as np


def make_layer_block(
    in_c: int,
    out_c: int,
    stride: int = 1,
    padding_type: int = 1,
    gain_weight: float = 1,
    gain_bias: float = 1,
):
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
            gain_weight=gain_weight,
            gain_bias=gain_bias,
        ),
        MixtureReLU(),
        BatchNorm2d(out_c),
        Conv2d(
            out_c,
            out_c,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_weight,
            gain_bias=gain_bias,
        ),
        MixtureReLU(),
        BatchNorm2d(out_c),
    )


def resnet18_cifar10(gain_w: float = 1, gain_b: float = 1) -> Sequential:
    """Resnet18 architecture for cifar10"""
    # 32x32
    initial_layers = [
        Conv2d(
            3,
            64,
            3,
            bias=False,
            padding=1,
            in_width=32,
            in_height=32,
            gain_weight=gain_w,
        ),
        MixtureReLU(),
        BatchNorm2d(64),
    ]

    resnet_layers = [
        # 32x32
        ResNetBlock(make_layer_block(64, 64, gain_weight=gain_w)),
        ResNetBlock(make_layer_block(64, 64, gain_weight=gain_w)),
        # 16x16
        ResNetBlock(
            make_layer_block(64, 128, 2, 2, gain_weight=gain_w),
            LayerBlock(
                Conv2d(
                    64,
                    128,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                ),
                MixtureReLU(),
                BatchNorm2d(128),
            ),
        ),
        ResNetBlock(make_layer_block(128, 128, gain_weight=gain_w)),
        # 8x8
        ResNetBlock(
            make_layer_block(128, 256, 2, 2, gain_weight=gain_w),
            LayerBlock(
                Conv2d(
                    128,
                    256,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                ),
                MixtureReLU(),
                BatchNorm2d(256),
            ),
        ),
        ResNetBlock(make_layer_block(256, 256, gain_weight=gain_w)),
        # 4x4
        ResNetBlock(
            make_layer_block(256, 512, 2, 2, gain_weight=gain_w),
            LayerBlock(
                Conv2d(
                    256,
                    512,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                ),
                MixtureReLU(),
                BatchNorm2d(512),
            ),
        ),
        ResNetBlock(make_layer_block(512, 512, gain_weight=gain_w)),
    ]

    final_layers = [
        AvgPool2d(4),
        Linear(512, 11, gain_weight=gain_w, gain_bias=gain_b),
    ]

    return Sequential(*initial_layers, *resnet_layers, *final_layers)


def resnet18_imagenet(
    gain_w: float = 1, gain_b: float = 1, nb_outputs=1001
) -> Sequential:
    """Resnet18 architecture for imagenet"""
    # 224x224
    initial_layers = [
        Conv2d(
            3,
            64,
            6,
            bias=False,
            padding=2,
            stride=2,
            in_width=224,
            in_height=224,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        MixtureReLU(),
        BatchNorm2d(64),
        AvgPool2d(3, stride=2, padding=1, padding_type=2),
    ]

    resnet_layers = [
        # 56x56
        ResNetBlock(
            make_layer_block(64, 64, gain_weight=gain_w, gain_bias=gain_b)
        ),
        ResNetBlock(
            make_layer_block(64, 64, gain_weight=gain_w, gain_bias=gain_b)
        ),
        # 28x28
        ResNetBlock(
            make_layer_block(64, 128, 2, 2, gain_weight=gain_w),
            LayerBlock(
                Conv2d(
                    64,
                    128,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                    gain_bias=gain_b,
                ),
                MixtureReLU(),
                BatchNorm2d(128),
            ),
        ),
        ResNetBlock(
            make_layer_block(128, 128, gain_weight=gain_w, gain_bias=gain_b)
        ),
        # 14x14
        ResNetBlock(
            make_layer_block(
                128, 256, 2, 2, gain_weight=gain_w, gain_bias=gain_b
            ),
            LayerBlock(
                Conv2d(
                    128,
                    256,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                    gain_bias=gain_b,
                ),
                MixtureReLU(),
                BatchNorm2d(256),
            ),
        ),
        ResNetBlock(
            make_layer_block(256, 256, gain_weight=gain_w, gain_bias=gain_b)
        ),
        # 7x7
        ResNetBlock(
            make_layer_block(
                256, 512, 2, 2, gain_weight=gain_w, gain_bias=gain_b
            ),
            LayerBlock(
                Conv2d(
                    256,
                    512,
                    2,
                    bias=False,
                    stride=2,
                    gain_weight=gain_w,
                    gain_bias=gain_b,
                ),
                MixtureReLU(),
                BatchNorm2d(512),
            ),
        ),
        ResNetBlock(
            make_layer_block(512, 512, gain_weight=gain_w, gain_bias=gain_b)
        ),
    ]

    final_layers = [
        AvgPool2d(7),
        Linear(
            512, nb_outputs, gain_weight=gain_w, gain_bias=gain_b, bias=True
        ),
    ]

    return Sequential(*initial_layers, *resnet_layers, *final_layers)
