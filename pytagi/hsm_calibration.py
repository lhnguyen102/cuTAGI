"""Post-hoc gain calibration of the hierarchical softmax output.

A hierarchical softmax classifier reads its class probability off a binary
tree. At node h on the path of class k, the network output Z_h gives the branch
probability P_h = Phi(s_h * G_h * Z_h), where Phi is the standard normal CDF,
s_h = +1 or -1 is the branch sign and G_h is a readout gain. The class
probability is the product of P_h over the path.

A network trained with a small observation noise is accurate but
overconfident. This module keeps the trained network fixed and treats each
gain G_h as a Gaussian random variable. It updates the gains on a held-out
split with an approximate Bayesian update, using the observation "the branch
of the true class was taken" with the Bernoulli variance mu_P (1 - mu_P).

The mean and the variance of P_h use Gaussian moment matching of the product
G_h * Z_h. The covariance of G_h and P_h is var_G * d mu_P / d mu_G (Stein's
lemma applied to the moment-matched mean), which agrees with Monte Carlo
better than the covariance of a joint Gaussian approximation of (G_h, G_h Z_h).
The update is therefore a scalar recursion per node that needs the
forward-pass means and variances only. Neither the training loop nor the
C++/CUDA backend is involved.

The fixed readout of the library is this model with G_h = 3 and no gain
uncertainty, so the default prior (mean 3) starts from the fixed readout.

When the number of classes is not a power of two, the tree has missing
leaves. The predictive class probabilities are then normalized over the
classes, while the calibration update uses each node of the path of the true
class on its own, so its objective differs slightly from the normalized
predictive.

Typical use::

    calibrator = HSMGainCalibrator(num_classes=10, sharing="level")
    calibrator.calibrate(mu_val, var_val, labels_val)
    prob, prob_var = calibrator.class_probabilities(mu_test, var_test)
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

import numpy as np

from pytagi.tagi_utils import Utils

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS = (
    np.polynomial.legendre.leggauss(8)
)

# numpy has no erfc. scipy is not a dependency of pytagi, so it is used only
# when it is installed (about 50 times faster than the fallback on large
# arrays).
try:
    from scipy.special import erfc as _erfc
except ImportError:  # pragma: no cover
    _erfc = np.vectorize(math.erfc, otypes=[np.float64])

# Number of array elements processed at once when computing node moments.
_CHUNK_ELEMENTS = 1 << 18
# Gain updates evaluated together only when at least this many are independent.
_MIN_VECTOR_WIDTH = 6


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-np.asarray(x, dtype=np.float64) / _SQRT2)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2)


def _phi_variance(a: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Variance of Phi(X) for a Gaussian X with variance ``var``.

    ``a`` is the standardized mean mu_X / sqrt(1 + var). The variance is the
    integral of exp(-a^2 (1 + t^2) / 2) / (1 + t^2) over t in [b, 1], divided by
    pi, with b = 1 / sqrt(1 + 2 var). It is evaluated with an 8-point
    Gauss-Legendre rule, which stays positive and avoids the cancellation of
    the closed form with Owen's T function for small ``var``.
    """
    b = 1.0 / np.sqrt(1.0 + 2.0 * var)
    half = 0.5 * (1.0 - b)
    mid = 0.5 * (1.0 + b)
    exponent = -0.5 * a * a
    total = np.zeros_like(a)
    for node, weight in zip(_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS):
        one_plus_t2 = 1.0 + np.square(mid + half * node)
        total += weight * np.exp(exponent * one_plus_t2) / one_plus_t2
    return (half / math.pi) * total


def _branch_mean_var(mu_z, var_z, mu_gain, var_gain, with_variance=False):
    """Means of both branches, Phi(a) and Phi(-a), and the variance of
    Phi(G * Z): the part of ``_branch_moments`` that prediction needs.

    The smaller of the two means is evaluated directly, so a branch
    probability in the far tail does not round to zero as 1 - Phi(a) would.

    :return: ``(mu_plus, mu_minus, var)``; ``var`` is None unless
        ``with_variance``.
    """
    var_s = var_gain * var_z + var_gain * mu_z**2 + mu_gain**2 * var_z
    a = mu_gain * mu_z / np.sqrt(1.0 + var_s)
    var_p = _phi_variance(a, var_s) if with_variance else None
    tail = _normal_cdf(-np.abs(a))
    positive = a >= 0.0
    mu_plus = np.where(positive, 1.0 - tail, tail)
    mu_minus = np.where(positive, tail, 1.0 - tail)
    return mu_plus, mu_minus, var_p


def _branch_moments(
    mu_z, var_z, mu_gain, var_gain, sign, with_variance: bool = False
):
    """Moments of the branch probability P = Phi(sign * G * Z).

    G and Z are independent Gaussians. The mean and the variance of P replace
    the product S = G * Z by a Gaussian with its exact mean and variance,
    which is exact when either G or Z is deterministic. The covariance is
    var_G * d mu_P / d mu_G, from Stein's lemma.

    :return: The mean of P, the covariance of G and P, and the variance of P
        (None unless ``with_variance``).
    """
    mu_s = mu_gain * mu_z
    var_s = var_gain * var_z + var_gain * mu_z**2 + mu_gain**2 * var_z
    one_plus_var_s = 1.0 + var_s
    a = sign * mu_s / np.sqrt(one_plus_var_s)
    mu_p = _normal_cdf(a)
    cov_gain_p = (
        sign
        * var_gain
        * mu_z
        * (one_plus_var_s - mu_gain**2 * var_z)
        / one_plus_var_s**1.5
        * _normal_pdf(a)
    )
    var_p = _phi_variance(a, var_s) if with_variance else None
    return mu_p, cov_gain_p, var_p


class HSMGainCalibrator:
    """Gaussian readout gains for the hierarchical softmax.

    The branch probability Phi(s * G * Z) has its mean and variance from
    Gaussian moment matching of G * Z, and its covariance with G from Stein's
    lemma (see the module documentation).

    :param num_classes: The number of classes.
    :type num_classes: int
    :param sharing: ``"node"`` for one gain per tree node, ``"level"`` for one
        gain per tree level, or ``"global"`` for a single gain.
    :type sharing: str
    :param mu_gain_init: Prior mean of the gains. The default 3 is the gain of
        the fixed readout (``Utils.obs_to_label_prob``), so the prior is "no
        change from the fixed readout".
    :type mu_gain_init: float
    :param sigma_gain_init: Prior standard deviation of the gains.
    :type sigma_gain_init: float
    :param min_mu_gain: Lower bound of the gain mean. A negative gain would swap
        the two branches of a node.
    :type min_mu_gain: float
    :param eps: The branch probability is clipped to [eps, 1 - eps] before the
        update to keep the Bernoulli variance positive.
    :type eps: float
    """

    def __init__(
        self,
        num_classes: int,
        sharing: str = "level",
        mu_gain_init: float = 3.0,
        sigma_gain_init: float = 1.0,
        min_mu_gain: float = 1e-3,
        eps: float = 1e-6,
    ):
        if sharing not in ("node", "level", "global"):
            raise ValueError(
                f"sharing must be 'node', 'level' or 'global', got {sharing!r}"
            )
        hrc = Utils().get_hierarchical_softmax(num_classes)
        self.num_classes = num_classes
        self.num_levels = int(hrc.num_obs)
        self.num_units = int(hrc.len)
        self.min_mu_gain = min_mu_gain
        self.eps = eps

        # Row k lists the signs and the output units on the path of class k.
        # The library numbers output units from 1.
        shape = (num_classes, self.num_levels)
        self._signs = np.asarray(hrc.obs, dtype=np.float64).reshape(shape)
        self._path_units = (
            np.asarray(hrc.idx, dtype=np.int64).reshape(shape) - 1
        )

        if sharing == "node":
            unit_to_gain = np.arange(self.num_units)
        elif sharing == "level":
            unit_to_gain = np.zeros(self.num_units, dtype=np.int64)
            for level in range(self.num_levels):
                unit_to_gain[self._path_units[:, level]] = level
        else:
            unit_to_gain = np.zeros(self.num_units, dtype=np.int64)
        self._unit_to_gain = unit_to_gain
        self._path_gains = unit_to_gain[self._path_units]

        self._build_tree()

        num_gains = int(unit_to_gain.max()) + 1
        self._var_gain_init = sigma_gain_init**2
        self.mu_gain = np.full(num_gains, mu_gain_init, dtype=np.float64)
        self.var_gain = np.full(
            num_gains, self._var_gain_init, dtype=np.float64
        )

    @property
    def sigma_gain(self) -> np.ndarray:
        """Standard deviation of the gains."""
        return np.sqrt(self.var_gain)

    def _build_tree(self) -> None:
        """Derive the tree from the class paths.

        ``_levels`` lists, for every level below the root, the units on it with
        their parent and the branch taken from the parent. This gives the
        probability of every path prefix with one pass over the units instead
        of one pass over the (class, level) pairs. ``_ragged`` describes the
        units whose subtree is missing some class, the only ones that make the
        class probabilities sum to less than one.
        """
        units, signs = self._path_units, self._signs
        num_levels = self.num_levels
        self._root = int(units[0, 0])
        self._levels = []
        for level in range(1, num_levels):
            child, first = np.unique(units[:, level], return_index=True)
            self._levels.append(
                (child, units[first, level - 1], signs[first, level - 1] > 0.0)
            )

        leaf = -2
        child_of = np.full((self.num_units, 2), -1, dtype=np.int64)
        for level in range(num_levels):
            below = units[:, level + 1] if level + 1 < num_levels else leaf
            child_of[units[:, level], (signs[:, level] < 0.0).astype(int)] = (
                below
            )
        complete = np.zeros(self.num_units, dtype=bool)
        groups = []
        for level in range(num_levels - 1, -1, -1):
            nodes = np.unique(units[:, level])
            full = np.ones(nodes.shape[0], dtype=bool)
            for branch in range(2):
                child = child_of[nodes, branch]
                full &= (child == leaf) | (
                    (child >= 0) & complete[np.maximum(child, 0)]
                )
            complete[nodes] = full
            groups.append(nodes[~full])
        self._root_complete = bool(complete[self._root])

        # Incomplete units, deepest level first, with the position of each
        # child in the dynamic program (see ``_subtree_mass``).
        self._ragged_units = np.concatenate(groups)
        num_ragged = self._ragged_units.shape[0]
        position = np.full(self.num_units, -1, dtype=np.int64)
        position[self._ragged_units] = np.arange(num_ragged)

        def code_of(child):
            """Column of a child subtree in the table of ``_subtree_mass``."""
            known = np.maximum(child, 0)
            code = np.where(child == -1, num_ragged, position[known])
            whole = (child == leaf) | ((child >= 0) & complete[known])
            return np.where(whole, num_ragged + 1, code)

        self._ragged_groups = []
        start = 0
        for nodes in groups:
            self._ragged_groups.append(
                (
                    start,
                    start + nodes.shape[0],
                    code_of(child_of[nodes, 0]),
                    code_of(child_of[nodes, 1]),
                )
            )
            start += nodes.shape[0]
        self._root_position = int(position[self._root])
        # Subtree on the other side of the path of class k at each level.
        other = (signs > 0.0).astype(int)
        self._sibling_codes = code_of(child_of[units, other])

    def calibrate(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        labels: Sequence[int],
        batch_size: int = 4096,
        reset_variance: bool = True,
    ) -> None:
        """Update the gains with one pass over a held-out split.

        Use data that was not used to train the network.

        The samples are processed in order, as a sequence of Bayesian updates.
        Inside a mini-batch, updates of different gains do not depend on each
        other, so they are evaluated together with array operations. The result
        does not depend on ``batch_size``.

        :param mu_z: Output means of the network, shape (N, len) or flat.
        :type mu_z: np.ndarray
        :param var_z: Output variances of the network, same shape.
        :type var_z: np.ndarray
        :param labels: The true class of each of the N samples, integers in
            [0, num_classes).
        :type labels: Sequence[int]
        :param batch_size: Samples gathered at a time. Only the units on the
            path of the label are read, so float32 outputs are not copied.
        :type batch_size: int
        :param reset_variance: Reset the gain variance to the prior first. Pass
            False to stream a split through several calls.
        :type reset_variance: bool
        :raises ValueError: If the shapes or the labels are invalid, or if an
            output read on the path of a label is not finite or has a
            negative variance.
        """
        mu_z = np.asarray(mu_z).reshape(-1, self.num_units)
        var_z = np.asarray(var_z).reshape(-1, self.num_units)
        if var_z.shape != mu_z.shape:
            raise ValueError(
                f"var_z has shape {var_z.shape}, mu_z has shape {mu_z.shape}"
            )
        labels = self._check_labels(labels, mu_z.shape[0])
        if reset_variance:
            self.var_gain[:] = self._var_gain_init
        for start in range(0, labels.shape[0], batch_size):
            stop = start + batch_size
            self._update_batch(
                mu_z[start:stop], var_z[start:stop], labels[start:stop]
            )

    def _check_labels(self, labels, num_rows: int) -> np.ndarray:
        """Labels as an int64 array of length ``num_rows`` in range."""
        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != num_rows:
            raise ValueError(
                f"{num_rows} output rows for {labels.shape[0]} labels"
            )
        if labels.dtype.kind not in "iu":
            if labels.dtype.kind != "f" or not np.all(
                np.isfinite(labels) & (labels == np.round(labels))
            ):
                raise ValueError("labels must be integers")
        labels = labels.astype(np.int64)
        if labels.size and (
            labels.min() < 0 or labels.max() >= self.num_classes
        ):
            raise ValueError(
                f"labels must be in [0, {self.num_classes}), got"
                f" [{labels.min()}, {labels.max()}]"
            )
        return labels

    def _update_batch(self, mu_z, var_z, labels) -> None:
        units = self._path_units[labels]
        rows = np.arange(labels.shape[0])[:, None]
        mu_path = mu_z[rows, units].astype(np.float64).reshape(-1)
        var_path = var_z[rows, units].astype(np.float64).reshape(-1)
        if not (np.all(np.isfinite(mu_path)) and np.all(np.isfinite(var_path))):
            raise ValueError("non-finite network outputs")
        if np.any(var_path < 0.0):
            raise ValueError("negative output variances")
        sign = self._signs[labels].reshape(-1)
        gain = self._path_gains[labels].reshape(-1)

        # Update number r of a gain is evaluated in round r. The entries are
        # in sample order, so the order of the updates of each gain is kept.
        by_gain = np.argsort(gain, kind="stable")
        gain_sorted = gain[by_gain]
        first = np.flatnonzero(np.r_[True, gain_sorted[1:] != gain_sorted[:-1]])
        group = np.cumsum(np.r_[False, gain_sorted[1:] != gain_sorted[:-1]])
        rank = np.empty_like(by_gain)
        rank[by_gain] = np.arange(by_gain.shape[0]) - first[group]
        order = np.argsort(rank, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(rank))]
        mu_path, var_path = mu_path[order], var_path[order]
        sign, gain = sign[order], gain[order]

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo >= _MIN_VECTOR_WIDTH:
                self._update_gains(
                    gain[lo:hi], mu_path[lo:hi], var_path[lo:hi], sign[lo:hi]
                )
            else:
                for g, mz, vz, s in zip(
                    gain[lo:hi].tolist(),
                    mu_path[lo:hi].tolist(),
                    var_path[lo:hi].tolist(),
                    sign[lo:hi].tolist(),
                ):
                    self._update_gain_scalar(g, mz, vz, s)

    def _update_gains(self, gain, mu_z, var_z, sign) -> None:
        """Bayesian update of distinct gains, one observation each."""
        mu_gain, var_gain = self.mu_gain[gain], self.var_gain[gain]
        mu_p, cov_gain_p, _ = _branch_moments(
            mu_z, var_z, mu_gain, var_gain, sign
        )
        mu_p = np.clip(mu_p, self.eps, 1.0 - self.eps)
        var_obs = mu_p * (1.0 - mu_p)
        # The observation is 1: the branch of the true class was taken.
        mu_post = mu_gain + cov_gain_p / mu_p
        var_post = var_gain - cov_gain_p**2 / var_obs
        self.mu_gain[gain] = np.maximum(mu_post, self.min_mu_gain)
        self.var_gain[gain] = np.maximum(var_post, 0.0)

    def _update_gain_scalar(self, gain, mu_z, var_z, sign) -> None:
        """Same as ``_update_gains`` for one gain, without numpy overhead."""
        mu_gain, var_gain = float(self.mu_gain[gain]), float(
            self.var_gain[gain]
        )
        var_s = var_gain * var_z + var_gain * mu_z**2 + mu_gain**2 * var_z
        one_plus_var_s = 1.0 + var_s
        root = math.sqrt(one_plus_var_s)
        a = sign * mu_gain * mu_z / root
        mu_p = 0.5 * math.erfc(-a / _SQRT2)
        cov_gain_p = (
            sign
            * var_gain
            * mu_z
            * (one_plus_var_s - mu_gain**2 * var_z)
            / (one_plus_var_s * root)
            * _INV_SQRT_2PI
            * math.exp(-0.5 * a * a)
        )
        mu_p = min(max(mu_p, self.eps), 1.0 - self.eps)
        var_obs = mu_p * (1.0 - mu_p)
        self.mu_gain[gain] = max(mu_gain + cov_gain_p / mu_p, self.min_mu_gain)
        self.var_gain[gain] = max(var_gain - cov_gain_p**2 / var_obs, 0.0)

    def class_probabilities(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        compute_var: bool = True,
        classes: Optional[Sequence[int]] = None,
        num_threads: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Class probabilities computed with the calibrated gains.

        The probabilities sum to one when every leaf of the tree is a class.
        Otherwise (for example when the number of classes is not a power of
        two) they are normalized over the classes: p_k = u_k / M, where u_k
        is the product over the path of class k and M the sum over the
        classes. The variance treats the tree nodes as independent. For
        normalized probabilities it is the first-order delta method,
        (var(u_k) - 2 p_k cov(u_k, M) + p_k^2 var(M)) / E[M]^2, which keeps
        the covariance of u_k with M (the plain var(u_k) / E[M]^2
        overestimates the variance of the most probable classes).

        With a large tree, ask only for the classes you need: the cost is then
        proportional to the number of requested classes instead of the size of
        the tree for the paths, and to the number of incomplete subtrees for
        the normalization. See also ``label_probabilities`` and
        ``top_k_probabilities``.

        :param mu_z: Output means of the network, shape (N, len) or (len,).
        :type mu_z: np.ndarray
        :param var_z: Output variances of the network, same shape.
        :type var_z: np.ndarray
        :param compute_var: Also return the variance of the class probability.
        :type compute_var: bool
        :param classes: Classes to return, shape (C,) for all samples or (N, C)
            for one list per sample. Default: all classes.
        :type classes: Sequence[int]
        :param num_threads: Threads that process blocks of rows in parallel.
            The result does not depend on it.
        :type num_threads: int
        :return: The mean and the variance of the class probabilities, each of
            shape (N, num_classes) or (N, C). The variance is None if
            ``compute_var`` is False.
        :rtype: tuple
        """
        mu_z = np.asarray(mu_z).reshape(-1, self.num_units)
        var_z = np.asarray(var_z).reshape(-1, self.num_units)
        num_rows = mu_z.shape[0]
        if classes is not None:
            classes = np.asarray(classes, dtype=np.int64)
            if (
                classes.min(initial=0) < 0
                or classes.max(initial=0) >= self.num_classes
            ):
                raise ValueError("classes out of range")
            classes = np.broadcast_to(
                (
                    classes.reshape(-1, classes.shape[-1])
                    if classes.ndim > 1
                    else classes[None, :]
                ),
                (num_rows, classes.shape[-1]),
            )
        width = (
            self.num_units
            if classes is None
            else classes.shape[1] * self.num_levels
        )
        step = max(1, _CHUNK_ELEMENTS // max(width, 1))
        num_out = self.num_classes if classes is None else classes.shape[1]
        mu_class = np.empty((num_rows, num_out))
        var_class = np.empty((num_rows, num_out)) if compute_var else None

        def run(lo):
            sl = slice(lo, lo + step)
            if classes is None:
                out = self._all_classes(mu_z[sl], var_z[sl], compute_var)
            else:
                out = self._some_classes(
                    mu_z[sl], var_z[sl], classes[sl], compute_var
                )
            mu_class[sl] = out[0]
            if compute_var:
                var_class[sl] = out[1]

        starts = range(0, num_rows, step)
        if num_threads > 1:
            with ThreadPoolExecutor(num_threads) as pool:
                list(pool.map(run, starts))
        else:
            for lo in starts:
                run(lo)
        return mu_class, var_class

    def label_probabilities(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        labels: Sequence[int],
        compute_var: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Probability of the given class of each sample, shape (N,).

        This is what the negative log-likelihood needs. It does not build the
        (N, num_classes) matrix. Same values as ``class_probabilities``.

        :raises ValueError: If ``labels`` are not N integers in
            [0, num_classes).
        """
        mu_z = np.asarray(mu_z).reshape(-1, self.num_units)
        labels = self._check_labels(labels, mu_z.shape[0]).reshape(-1, 1)
        mu_class, var_class = self.class_probabilities(
            mu_z, var_z, compute_var, classes=labels
        )
        return mu_class[:, 0], None if var_class is None else var_class[:, 0]

    def top_k_probabilities(
        self,
        mu_z: np.ndarray,
        var_z: np.ndarray,
        k: int,
        num_threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The ``k`` most probable classes of each sample.

        The result is exact. The full matrix is built a few rows at a time and
        not kept.

        :param k: Number of classes to return, 1 <= k <= num_classes.
        :type k: int
        :param num_threads: Passed to ``class_probabilities``.
        :type num_threads: int
        :return: The probabilities (N, k) in decreasing order and the class
            indices (N, k).
        """
        if not 1 <= k <= self.num_classes:
            raise ValueError(f"k must be in [1, {self.num_classes}], got {k}")
        mu_z = np.asarray(mu_z).reshape(-1, self.num_units)
        var_z = np.asarray(var_z).reshape(-1, self.num_units)
        # Each call gets one block of rows per thread.
        step = max(1, _CHUNK_ELEMENTS // max(self.num_units, 1))
        step *= max(1, num_threads)
        top_prob = np.empty((mu_z.shape[0], k))
        top_class = np.empty((mu_z.shape[0], k), dtype=np.int64)
        for lo in range(0, mu_z.shape[0], step):
            prob, _ = self.class_probabilities(
                mu_z[lo : lo + step],
                var_z[lo : lo + step],
                compute_var=False,
                num_threads=num_threads,
            )
            part = np.argpartition(-prob, k - 1, axis=1)[:, :k]
            part_prob = np.take_along_axis(prob, part, axis=1)
            rank = np.argsort(-part_prob, axis=1)
            top_class[lo : lo + step] = np.take_along_axis(part, rank, axis=1)
            top_prob[lo : lo + step] = np.take_along_axis(
                part_prob, rank, axis=1
            )
        return top_prob, top_class

    def _all_classes(self, mu_z, var_z, compute_var):
        """All classes: path products by one pass over the tree units."""
        mu_z = mu_z.astype(np.float64, copy=False)
        var_z = var_z.astype(np.float64, copy=False)
        gain = self._unit_to_gain
        mu_p, mu_q, var_p = _branch_mean_var(
            mu_z, var_z, self.mu_gain[gain], self.var_gain[gain], compute_var
        )
        prefix = np.zeros_like(mu_p)  # product of the branch means above a unit
        prefix[:, self._root] = 1.0
        if compute_var:
            prefix_sq_var = np.zeros_like(mu_p)  # product of var + mean^2
            prefix_sq = np.zeros_like(mu_p)  # product of mean^2
            prefix_sq_var[:, self._root] = 1.0
            prefix_sq[:, self._root] = 1.0
        for child, parent, positive in self._levels:
            branch = np.where(positive, mu_p[:, parent], mu_q[:, parent])
            prefix[:, child] = prefix[:, parent] * branch
            if compute_var:
                prefix_sq_var[:, child] = prefix_sq_var[:, parent] * (
                    var_p[:, parent] + branch**2
                )
                prefix_sq[:, child] = prefix_sq[:, parent] * branch**2
        last = self._path_units[:, -1]
        branch = np.where(
            self._signs[:, -1] > 0.0, mu_p[:, last], mu_q[:, last]
        )
        mu_class = prefix[:, last] * branch
        var_class = None
        if compute_var:
            var_class = (
                prefix_sq_var[:, last] * (var_p[:, last] + branch**2)
                - prefix_sq[:, last] * branch**2
            )
        if self._root_complete:
            return mu_class, var_class
        if not compute_var:
            return mu_class / mu_class.sum(axis=-1, keepdims=True), None
        positive = self._signs > 0.0
        units = self._path_units
        mu_path = np.where(positive, mu_p[:, units], mu_q[:, units])
        mu_other = np.where(positive, mu_q[:, units], mu_p[:, units])
        return self._normalize(
            mu_z,
            var_z,
            mu_class,
            var_class,
            mu_path,
            mu_other,
            var_p[:, units],
            self._sibling_codes,
        )

    def _some_classes(self, mu_z, var_z, classes, compute_var):
        """Requested classes only: gather the units on their paths."""
        units = self._path_units[classes]  # (N, C, levels)
        rows = np.arange(mu_z.shape[0])[:, None, None]
        gain = self._path_gains[classes]
        mu_p, mu_q, var_p = _branch_mean_var(
            mu_z[rows, units].astype(np.float64),
            var_z[rows, units].astype(np.float64),
            self.mu_gain[gain],
            self.var_gain[gain],
            compute_var,
        )
        positive = self._signs[classes] > 0.0
        mu_path = np.where(positive, mu_p, mu_q)
        mu_class = np.prod(mu_path, axis=-1)
        var_class = None
        if compute_var:
            var_class = np.prod(var_p + mu_path**2, axis=-1) - np.prod(
                mu_path**2, axis=-1
            )
        if self._root_complete:
            return mu_class, var_class
        if not compute_var:
            mass = self._subtree_mass(mu_z, var_z)[0]
            return mu_class / mass[:, None], None
        return self._normalize(
            mu_z,
            var_z,
            mu_class,
            var_class,
            mu_path,
            np.where(positive, mu_q, mu_p),
            var_p,
            self._sibling_codes[classes],
        )

    def _normalize(
        self, mu_z, var_z, mu_u, var_u, mu_path, mu_other, var_path, siblings
    ):
        """Normalized probabilities p_k = u_k / M and their variance.

        The variance is the first-order delta method,
        (var(u_k) - 2 p_k cov(u_k, M) + p_k^2 var(M)) / E[M]^2, with the nodes
        independent. cov(u_k, M) = E[u_k M] - E[u_k] E[M], and E[u_k M] is a
        recursion up the path of class k: at a node where the path takes the
        branch B, E[B^2] E[u M] of the child on the path plus E[B B'] E[u]
        E[M] of the two children, where B' = 1 - B is the other branch.

        ``mu_path``, ``mu_other`` and ``var_path`` have shape (..., levels);
        ``siblings`` gives the column of the other child subtree of each node
        in the table of ``_subtree_mass``.
        """
        mass, mass_table, second = self._subtree_mass(mu_z, var_z, True)
        rows = np.arange(mu_z.shape[0]).reshape((-1,) + (1,) * siblings.ndim)
        if siblings.ndim == 3:  # one list of classes per row
            rows = rows[..., 0]
        sibling_mass = mass_table[rows, siblings]
        product = np.ones_like(mu_u)  # E[u] below the current node
        joint = np.ones_like(mu_u)  # E[u M] below the current node
        for level in range(self.num_levels - 1, -1, -1):
            mean = mu_path[..., level]
            var = var_path[..., level]
            joint = (var + mean**2) * joint + (
                mean * mu_other[..., level] - var
            ) * product * sibling_mass[..., level]
            product = product * mean
        mass = mass[:, None]
        prob = mu_u / mass
        cov = joint - mu_u * mass
        var_mass = second[:, None] - mass**2
        var = (var_u - 2.0 * prob * cov + prob**2 * var_mass) / mass**2
        return prob, np.maximum(var, 0.0)

    def _subtree_mass(self, mu_z, var_z, second_moment=False):
        """Sum M of the class probabilities, without visiting every class.

        Subtrees that contain all their classes sum to one. The sum of an
        incomplete subtree is P * (sum of the + subtree) + (1 - P) * (sum of
        the - subtree), evaluated from the deepest level up, with 1 - P
        computed as Phi(-a).

        :return: E[M] (N,); with ``second_moment``, also the table of E[M] of
            every incomplete subtree (followed by the columns 0 and 1 for a
            missing and a complete subtree) and E[M^2] (N,).
        """
        units = self._ragged_units
        gain = self._unit_to_gain[units]
        mu_p, mu_q, var_p = _branch_mean_var(
            mu_z[:, units].astype(np.float64),
            var_z[:, units].astype(np.float64),
            self.mu_gain[gain],
            self.var_gain[gain],
            second_moment,
        )
        total = np.empty((mu_z.shape[0], units.shape[0] + 2))
        total[:, -2] = 0.0
        total[:, -1] = 1.0
        if second_moment:
            square = total.copy()
        for start, stop, plus, minus in self._ragged_groups:
            p, q = mu_p[:, start:stop], mu_q[:, start:stop]
            total[:, start:stop] = p * total[:, plus] + q * total[:, minus]
            if second_moment:
                var = var_p[:, start:stop]
                square[:, start:stop] = (
                    (var + p**2) * square[:, plus]
                    + 2.0 * (p * q - var) * total[:, plus] * total[:, minus]
                    + (var + q**2) * square[:, minus]
                )
        root = self._root_position
        if not second_moment:
            return total[:, root], None, None
        return total[:, root], total, square[:, root]
