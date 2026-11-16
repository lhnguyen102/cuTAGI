# Temporary import. It will be removed in the final version
import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.hsm_calibration import HSMGainCalibrator
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)

NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)

CNN = Sequential(
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Conv2d(32, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Conv2d(64, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)


def make_loader(dataset, batch_size, train):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        collate_fn=collate,
    )


def collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images).numpy().reshape(-1), np.array(labels)


def negative_log_likelihood(prob, labels):
    true_prob = prob[np.arange(len(labels)), labels]
    return -np.mean(np.log(np.clip(true_prob, 1e-12, 1.0)))


def expected_calibration_error(prob, labels, num_bins=15):
    confidence = prob.max(axis=1)
    correct = prob.argmax(axis=1) == labels
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(
                correct[in_bin].mean() - confidence[in_bin].mean()
            )
    return ece


def forward_pass(net, loader, num_units):
    """Output means, variances and labels of a split, without any update."""
    mu_z, var_z, labels = [], [], []
    net.eval()
    for x, label in loader:
        m_pred, v_pred = net(x)
        mu_z.append(np.reshape(m_pred, (len(label), num_units)))
        var_z.append(np.reshape(v_pred, (len(label), num_units)))
        labels.append(label)
    return np.concatenate(mu_z), np.concatenate(var_z), np.concatenate(labels)


def report(name, prob, labels):
    accuracy = np.mean(prob.argmax(axis=1) == labels)
    print(
        f"{name:>12s} | accuracy {accuracy:.4f} | "
        f"NLL {negative_log_likelihood(prob, labels):.4f} | "
        f"ECE {expected_calibration_error(prob, labels):.4f}"
    )


def main(
    num_epochs: int = 5,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    num_calibration: int = 5000,
):
    """
    Train a hierarchical softmax classifier on CIFAR-10, then calibrate its
    readout gains on a held-out split and score the test set before and after.
    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    - sigma_v: float, observation noise of the output update
    - num_calibration: int, training images held out for calibration
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=False, download=True, transform=transform
    )
    # The last num_calibration images are never used to train the network.
    num_train = len(cifar_train) - num_calibration
    train_loader = make_loader(
        Subset(cifar_train, range(num_train)), batch_size, train=True
    )
    calibration_loader = make_loader(
        Subset(cifar_train, range(num_train, len(cifar_train))),
        batch_size,
        train=False,
    )
    test_loader = make_loader(cifar_test, batch_size, train=False)

    metric = HRCSoftmaxMetric(num_classes=10)
    num_units = metric.hrc_softmax.len
    utils = Utils()

    net = CNN
    if pytagi.cuda.is_available():
        net.to_device("cuda")
    else:
        net.set_threads(8)
    out_updater = OutputUpdater(net.device)
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        net.train()
        for x, labels in train_loader:
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)
            net(x)
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

    mu_cal, var_cal, labels_cal = forward_pass(
        net, calibration_loader, num_units
    )
    mu_test, var_test, labels_test = forward_pass(net, test_loader, num_units)

    # The fixed readout leaves out the missing leaves of the tree (16 leaves
    # for 10 classes), so its probabilities are normalized before scoring.
    prob_before = utils.obs_to_label_prob(
        mu_test, var_test, metric.hrc_softmax, 10
    )
    prob_before = prob_before / prob_before.sum(axis=1, keepdims=True)
    report("uncalibrated", prob_before, labels_test)

    calibrator = HSMGainCalibrator(10, sharing="level")
    calibrator.calibrate(mu_cal, var_cal, labels_cal)
    prob_after, _ = calibrator.class_probabilities(
        mu_test, var_test, compute_var=False
    )
    report("calibrated", prob_after, labels_test)
    print("gain means per level:", np.round(calibrator.mu_gain, 3))


if __name__ == "__main__":
    fire.Fire(main)
