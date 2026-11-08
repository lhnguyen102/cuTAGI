.. _hsm_calibration:

=========================================
Hierarchical Softmax Gain Calibration
=========================================

A classifier with a hierarchical softmax output trained with a small
observation noise (for example ``sigma_v = 0.05``) is accurate but
overconfident. ``pytagi.hsm_calibration.HSMGainCalibrator`` corrects the
confidence after training, without changing the network or the training loop.

Method
------

The probability of the branch taken at tree node :math:`h` is

.. math::

   P_h = \Phi(s_h \, G_h \, Z_h),

where :math:`Z_h` is the network output for node :math:`h`, :math:`s_h = \pm 1`
is the branch sign, :math:`\Phi` is the standard normal CDF and :math:`G_h` is a
readout gain. The class probability is the product of :math:`P_h` over the path
of the class.

Each gain is a Gaussian random variable with a prior mean and standard
deviation. For a held-out sample, the observation "the branch of the true class
was taken" updates the gain with the approximate Bayesian update of TAGI, using
the Bernoulli variance :math:`\mu_{P_h}(1 - \mu_{P_h})`. The mean and the
variance of :math:`P_h` use Gaussian moment matching of the product
:math:`G_h Z_h`. The covariance of :math:`G_h` and :math:`P_h` is
:math:`\sigma_{G_h}^2 \, \partial \mu_{P_h} / \partial \mu_{G_h}` (Stein's
lemma), which agrees with Monte Carlo better than the covariance of a joint
Gaussian approximation of :math:`(G_h, G_h Z_h)`. Calibration is therefore a
scalar recursion per node that reads only the forward-pass means and variances
of the network.

The fixed readout of ``Utils.obs_to_label_prob`` is this model with
:math:`G_h = 3` and no gain uncertainty. The default prior mean of the gains is
therefore 3 (standard deviation 1): before any data, the calibrated readout is
the fixed readout.

When the number of classes is not a power of two, the tree has missing leaves
and the class probabilities :math:`u_k` (products over the paths) sum to
:math:`M < 1`. The predictive probabilities are normalized,
:math:`p_k = u_k / M`, and their variance is the first-order delta method

.. math::

   \mathrm{var}(p_k) \approx \frac{\mathrm{var}(u_k)
   - 2 p_k \mathrm{cov}(u_k, M) + p_k^2 \mathrm{var}(M)}{\mathbb{E}[M]^2},

with the tree nodes treated as independent. Keeping the covariance of
:math:`u_k` with :math:`M` matters: :math:`\mathrm{var}(u_k) / \mathbb{E}[M]^2`
alone overestimates the variance of the most probable classes. The calibration
update uses each node of the path of the true class on its own, so for such
trees its objective differs slightly from the normalized predictive.

The gains can be shared by node (``sharing="node"``), by tree level
(``sharing="level"``, default) or globally (``sharing="global"``).

Usage
-----

Run the trained network on a split that was not used for training and pass its
output means and variances with the labels:

.. code-block:: python

   from pytagi.hsm_calibration import HSMGainCalibrator

   calibrator = HSMGainCalibrator(num_classes=10, sharing="level")
   calibrator.calibrate(mu_val, var_val, labels_val)
   prob, prob_var = calibrator.class_probabilities(mu_test, var_test)

``mu_val`` and ``var_val`` have shape ``(num_samples, len)``, where ``len`` is
the number of output units of the hierarchical softmax (11 for 10 classes).
``prob`` has shape ``(num_samples, num_classes)``.

A complete example that trains a CIFAR-10 classifier and reports accuracy,
negative log-likelihood (NLL) and expected calibration error (ECE) before and
after calibration is in ``examples/hsm_gain_calibration.py``.

Large trees
-----------

With a large vocabulary the matrix of all class probabilities is the expensive
part. ``class_probabilities`` accepts ``classes`` (a list of classes, or one
list per sample), ``label_probabilities(mu, var, labels)`` returns the
probability of the true class of each sample (what the NLL needs) and
``top_k_probabilities(mu, var, k)`` returns the most probable classes without
keeping the full matrix. These return the same values as the full matrix.
``calibrate`` processes the samples in order and accepts
``reset_variance=False`` to stream a split through several calls.
