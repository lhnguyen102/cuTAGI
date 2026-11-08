import math
import os
import sys
import unittest

import numpy as np

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.hsm_calibration import (
    HSMGainCalibrator,
    _branch_moments,
    _normal_cdf,
)


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


def sample_outputs(calibrator, num_samples, rng, scale=1.0):
    """Network outputs and labels drawn from the model P(class) with the
    calibrator's gain means. ``scale`` multiplies the outputs after the labels
    are drawn, which makes the network overconfident when above 1."""
    mu_z = rng.normal(0.0, 2.0, size=(num_samples, calibrator.num_units))
    var_z = np.full_like(mu_z, 0.05)
    prob, _ = calibrator.class_probabilities(mu_z, var_z, compute_var=False)
    cumulative = np.cumsum(prob, axis=1)
    draw = rng.uniform(size=(num_samples, 1)) * cumulative[:, -1:]
    labels = np.minimum(
        (draw > cumulative).sum(axis=1), calibrator.num_classes - 1
    )
    return mu_z * scale, var_z * scale**2, labels


def reference_calibrate(calibrator, mu_z, var_z, labels):
    """Sample-by-sample, node-by-node update that defines the semantics of
    ``HSMGainCalibrator.calibrate``."""
    calibrator.var_gain[:] = calibrator._var_gain_init
    for row, label in enumerate(labels):
        for level in range(calibrator.num_levels):
            unit = calibrator._path_units[label, level]
            gain = calibrator._path_gains[label, level]
            mu_p, cov, _ = _branch_moments(
                mu_z[row, unit],
                var_z[row, unit],
                calibrator.mu_gain[gain],
                calibrator.var_gain[gain],
                calibrator._signs[label, level],
            )
            mu_p = np.clip(mu_p, calibrator.eps, 1.0 - calibrator.eps)
            mu_post = calibrator.mu_gain[gain] + cov / mu_p
            var_post = calibrator.var_gain[gain] - cov**2 / (mu_p * (1 - mu_p))
            calibrator.mu_gain[gain] = max(mu_post, calibrator.min_mu_gain)
            calibrator.var_gain[gain] = max(var_post, 0.0)


def reference_probabilities(calibrator, mu_z, var_z):
    """Class probabilities by an explicit loop over the classes. The variance
    of the normalized probability u_k / Z is the delta method with the
    covariances of all pairs of path products."""
    gain = calibrator._unit_to_gain
    mu_p, _, var_p = _branch_moments(
        mu_z,
        var_z,
        calibrator.mu_gain[gain],
        calibrator.var_gain[gain],
        1.0,
        True,
    )
    num_classes = calibrator.num_classes
    branches = []  # {unit: sign} on the path of each class
    for k in range(num_classes):
        branches.append(
            dict(zip(calibrator._path_units[k], calibrator._signs[k]))
        )
    mean = np.ones((mu_z.shape[0], num_classes))
    for k, path in enumerate(branches):
        for unit, sign in path.items():
            mean[:, k] *= mu_p[:, unit] if sign > 0 else 1.0 - mu_p[:, unit]
    # E[u_k u_j]: the nodes are independent; a shared node contributes
    # E[P^2], E[(1 - P)^2] or E[P (1 - P)].
    second = np.ones((mu_z.shape[0], num_classes, num_classes))
    for k, path_k in enumerate(branches):
        for j, path_j in enumerate(branches):
            for unit in set(path_k) | set(path_j):
                p, v = mu_p[:, unit], var_p[:, unit]
                signs = {path.get(unit) for path in (path_k, path_j)}
                signs.discard(None)
                if unit in path_k and unit in path_j:
                    if signs == {1.0}:
                        second[:, k, j] *= v + p**2
                    elif signs == {-1.0}:
                        second[:, k, j] *= v + (1.0 - p) ** 2
                    else:
                        second[:, k, j] *= p * (1.0 - p) - v
                else:
                    second[:, k, j] *= p if signs == {1.0} else 1.0 - p
    cov = second - mean[:, :, None] * mean[:, None, :]
    total = mean.sum(axis=1, keepdims=True)
    prob = mean / total
    var_u = np.diagonal(cov, axis1=1, axis2=2)
    if calibrator._root_complete:
        return prob, var_u
    cov_total = cov.sum(axis=2)
    var_total = cov.sum(axis=(1, 2))[:, None]
    var = (var_u - 2 * prob * cov_total + prob**2 * var_total) / total**2
    return prob, var


class TestBranchMoments(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.mu_z, self.var_z = 0.8, 0.4
        self.mu_gain, self.var_gain = 1.2, 0.01
        n = 2_000_000
        self.gain = rng.normal(self.mu_gain, math.sqrt(self.var_gain), n)
        z = rng.normal(self.mu_z, math.sqrt(self.var_z), n)
        self.p = 0.5 * (
            1.0 + np.vectorize(math.erf)(self.gain * z / math.sqrt(2))
        )

    def test_moments_match_monte_carlo(self):
        mu_p, cov_gain_p, var_p = _branch_moments(
            self.mu_z,
            self.var_z,
            self.mu_gain,
            self.var_gain,
            1.0,
            with_variance=True,
        )
        self.assertAlmostEqual(float(mu_p), self.p.mean(), places=2)
        self.assertAlmostEqual(float(var_p), self.p.var(), places=2)
        cov = np.mean((self.gain - self.mu_gain) * (self.p - self.p.mean()))
        self.assertAlmostEqual(float(cov_gain_p), cov, places=3)

    def test_opposite_sign_gives_complementary_mean(self):
        args = (self.mu_z, self.var_z, self.mu_gain, self.var_gain)
        mu_plus, cov_plus, var_plus = _branch_moments(*args, 1.0, True)
        mu_minus, cov_minus, var_minus = _branch_moments(*args, -1.0, True)
        self.assertAlmostEqual(float(mu_plus + mu_minus), 1.0)
        self.assertAlmostEqual(float(cov_plus + cov_minus), 0.0)
        self.assertAlmostEqual(float(var_plus), float(var_minus))


class TestHSMGainCalibrator(unittest.TestCase):
    def test_level_gains_recover_the_true_gains(self):
        rng = np.random.default_rng(0)
        truth = HSMGainCalibrator(8, sharing="level")
        truth.mu_gain[:] = [0.9, 0.6, 0.4]
        truth.var_gain[:] = 0.0
        mu_z, var_z, labels = sample_outputs(truth, 6000, rng)

        calibrator = HSMGainCalibrator(8, sharing="level")
        calibrator.calibrate(mu_z, var_z, labels)

        np.testing.assert_allclose(calibrator.mu_gain, truth.mu_gain, atol=0.1)
        self.assertTrue(np.all(calibrator.sigma_gain < 1.0))

    def test_class_probabilities_sum_to_one(self):
        rng = np.random.default_rng(1)
        for num_classes in (8, 10):
            calibrator = HSMGainCalibrator(num_classes)
            mu_z = rng.normal(size=(5, calibrator.num_units))
            var_z = rng.uniform(0.01, 1.0, size=mu_z.shape)
            prob, var = calibrator.class_probabilities(mu_z, var_z)
            self.assertEqual(prob.shape, (5, num_classes))
            np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-12)
            self.assertTrue(np.all(var >= 0.0))

    def test_calibration_fixes_an_overconfident_network(self):
        rng = np.random.default_rng(2)
        truth = HSMGainCalibrator(8, sharing="level")
        truth.mu_gain[:] = 0.5
        truth.var_gain[:] = 0.0
        # Outputs 4 times too large: the model is overconfident for gain 1.
        mu_val, var_val, labels_val = sample_outputs(truth, 4000, rng, 4.0)
        mu_test, var_test, labels_test = sample_outputs(truth, 4000, rng, 4.0)

        uncalibrated = HSMGainCalibrator(
            8, mu_gain_init=1.0, sigma_gain_init=0.0
        )
        calibrated = HSMGainCalibrator(8)
        calibrated.calibrate(mu_val, var_val, labels_val)

        prob_before, _ = uncalibrated.class_probabilities(mu_test, var_test)
        prob_after, _ = calibrated.class_probabilities(mu_test, var_test)
        before = negative_log_likelihood(prob_before, labels_test)
        after = negative_log_likelihood(prob_after, labels_test)
        self.assertLess(after, before)
        self.assertLess(
            expected_calibration_error(prob_after, labels_test),
            expected_calibration_error(prob_before, labels_test),
        )

    def test_fast_paths_match_the_reference(self):
        rng = np.random.default_rng(3)
        for num_classes in (8, 10, 37):
            for sharing in ("level", "node", "global"):
                fast = HSMGainCalibrator(num_classes, sharing=sharing)
                slow = HSMGainCalibrator(num_classes, sharing=sharing)
                n = 400
                labels = rng.integers(0, num_classes, n)
                mu_z = rng.normal(0.0, 1.5, size=(n, fast.num_units))
                mu_z[
                    np.arange(n)[:, None], fast._path_units[labels]
                ] += fast._signs[labels]
                var_z = rng.uniform(0.01, 0.3, size=mu_z.shape)
                fast.calibrate(mu_z, var_z, labels, batch_size=97)
                reference_calibrate(slow, mu_z, var_z, labels)
                np.testing.assert_allclose(
                    fast.mu_gain, slow.mu_gain, rtol=0, atol=1e-9
                )
                np.testing.assert_allclose(
                    fast.var_gain, slow.var_gain, rtol=0, atol=1e-9
                )

                prob, var = fast.class_probabilities(mu_z, var_z)
                ref_prob, ref_var = reference_probabilities(fast, mu_z, var_z)
                np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=1e-9)
                np.testing.assert_allclose(var, ref_var, rtol=0, atol=1e-9)

                classes = rng.integers(0, num_classes, size=(n, 3))
                sub, sub_var = fast.class_probabilities(
                    mu_z, var_z, classes=classes
                )
                np.testing.assert_allclose(
                    sub, np.take_along_axis(ref_prob, classes, 1), atol=1e-9
                )
                np.testing.assert_allclose(
                    sub_var, np.take_along_axis(ref_var, classes, 1), atol=1e-9
                )
                label_prob, _ = fast.label_probabilities(mu_z, var_z, labels)
                np.testing.assert_allclose(
                    label_prob, ref_prob[np.arange(n), labels], atol=1e-9
                )
                top_prob, top_class = fast.top_k_probabilities(mu_z, var_z, 3)
                np.testing.assert_allclose(
                    top_prob, -np.sort(-ref_prob, axis=1)[:, :3], atol=1e-9
                )
                np.testing.assert_allclose(
                    np.take_along_axis(ref_prob, top_class, 1),
                    top_prob,
                    atol=1e-9,
                )

    def test_variance_matches_monte_carlo_with_missing_leaves(self):
        # 10 classes: 16 leaves, so the probabilities are normalized.
        rng = np.random.default_rng(5)
        calibrator = HSMGainCalibrator(10, sharing="node")
        num_units = calibrator.num_units
        calibrator.mu_gain[:] = rng.uniform(0.8, 1.5, num_units)
        calibrator.var_gain[:] = 0.0025
        mu_z = rng.normal(0.0, 1.5, size=(4, num_units))
        var_z = rng.uniform(0.01, 0.05, size=mu_z.shape)
        prob, var = calibrator.class_probabilities(mu_z, var_z)

        units, signs = calibrator._path_units, calibrator._signs
        num_draws = 100_000
        for row in range(mu_z.shape[0]):
            gain = rng.normal(
                calibrator.mu_gain,
                np.sqrt(calibrator.var_gain),
                size=(num_draws, num_units),
            )
            z = rng.normal(
                mu_z[row], np.sqrt(var_z[row]), size=(num_draws, num_units)
            )
            p = _normal_cdf(gain * z)[:, units]
            path = np.prod(np.where(signs > 0.0, p, 1.0 - p), axis=-1)
            sample = path / path.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(
                prob[row], sample.mean(axis=0), rtol=0.05, atol=2e-3
            )
            np.testing.assert_allclose(
                var[row], sample.var(axis=0), rtol=0.15, atol=2e-4
            )

    def test_far_tail_probabilities_stay_positive(self):
        for num_classes in (8, 10):
            calibrator = HSMGainCalibrator(
                num_classes, mu_gain_init=1.0, sigma_gain_init=0.0
            )
            mu_z = np.full((2, calibrator.num_units), 12.0)
            var_z = np.zeros_like(mu_z)
            # The class reached by the negative branch at every level.
            tail = int(np.argmin(calibrator._signs.sum(axis=1)))
            label_prob, _ = calibrator.label_probabilities(
                mu_z, var_z, [tail, tail]
            )
            prob, var = calibrator.class_probabilities(mu_z, var_z)
            self.assertTrue(np.all(label_prob > 0.0))
            self.assertTrue(np.all(np.isfinite(-np.log(label_prob))))
            np.testing.assert_allclose(prob[:, tail], label_prob, rtol=1e-9)
            self.assertTrue(np.all(prob > 0.0))
            self.assertTrue(np.all(var >= 0.0))

    def test_top_k_threads_and_bounds(self):
        rng = np.random.default_rng(6)
        calibrator = HSMGainCalibrator(37)
        mu_z = rng.normal(size=(50, calibrator.num_units))
        var_z = rng.uniform(0.01, 0.3, size=mu_z.shape)
        one = calibrator.top_k_probabilities(mu_z, var_z, 5)
        four = calibrator.top_k_probabilities(mu_z, var_z, 5, num_threads=4)
        np.testing.assert_array_equal(one[0], four[0])
        np.testing.assert_array_equal(one[1], four[1])
        for k in (0, 38):
            with self.assertRaises(ValueError):
                calibrator.top_k_probabilities(mu_z, var_z, k)

    def test_streaming_equals_one_call(self):
        rng = np.random.default_rng(4)
        one = HSMGainCalibrator(10)
        two = HSMGainCalibrator(10)
        mu_z = rng.normal(size=(300, one.num_units))
        var_z = rng.uniform(0.01, 0.3, size=mu_z.shape)
        labels = rng.integers(0, 10, 300)
        one.calibrate(mu_z, var_z, labels)
        two.calibrate(mu_z[:120], var_z[:120], labels[:120])
        two.calibrate(
            mu_z[120:], var_z[120:], labels[120:], reset_variance=False
        )
        np.testing.assert_allclose(one.mu_gain, two.mu_gain, atol=1e-12)
        np.testing.assert_allclose(one.var_gain, two.var_gain, atol=1e-12)

    def test_sharing_modes_set_the_number_of_gains(self):
        self.assertEqual(HSMGainCalibrator(8, sharing="global").mu_gain.size, 1)
        self.assertEqual(HSMGainCalibrator(8, sharing="level").mu_gain.size, 3)
        self.assertEqual(HSMGainCalibrator(8, sharing="node").mu_gain.size, 7)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            HSMGainCalibrator(8, sharing="layer")
        calibrator = HSMGainCalibrator(8)
        with self.assertRaises(ValueError):
            calibrator.calibrate(
                np.zeros((3, calibrator.num_units)),
                np.ones((3, calibrator.num_units)),
                [0, 1],
            )
        mu_z = np.zeros((3, calibrator.num_units))
        var_z = np.ones_like(mu_z)
        for labels in ([0, 1, 8], [0, -1, 2], [0.5, 1, 2]):
            with self.assertRaises(ValueError):
                calibrator.calibrate(mu_z, var_z, labels)
            with self.assertRaises(ValueError):
                calibrator.label_probabilities(mu_z, var_z, labels)
        with self.assertRaises(ValueError):
            calibrator.label_probabilities(mu_z, var_z, [0, 1])
        bad = mu_z.copy()
        bad[:, calibrator._path_units[0, 0]] = np.nan
        with self.assertRaises(ValueError):
            calibrator.calibrate(bad, var_z, [0, 1, 2])
        calibrator.calibrate(mu_z, var_z, np.array([0.0, 1.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
