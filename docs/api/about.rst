.. _about:

====================
About py/cuTAGI
====================

The core developments of py/cuTAGI have been made by Luong-Ha Nguyen, building upon
the theoretical work done at Polytechnique Montreal in collaboration with James-A. Goulet,
Bhargab Deka, Van-Dai Vuong, and Miquel Florensa. The project started in 2018 when,
from our background with large-scale state-space models, we foresaw that it would be
possible to perform analytical Bayesian inference in neural networks
(see below our first try at what would become TAGI).

.. figure:: ../_static/TAGI_2018.png
   :width: 80%
   :align: center
   :alt: TAGI initial trial in 2018

   TAGI initial trial in 2018

Following the early proofs of concepts with small-scale examples with MLPs,
we slowly expanded the development of TAGI for CNN, autoencoders and GANs architectures.
Then came proofs of concepts with reinforcement learning toy problems which led
to full-scale applications on the Atari and MuJoCo benchmarks. The expansion of
TAGI's applicability to new architectures continued with LSTM networks along with
unprecedented features with analytical uncertainty quantification for Bayesian
neural networks, analytical adversarial attacks, inference-based optimization,
and general-purpose latent-space inference.

Despite our repeated successes at leveraging analytical inference in neural
networks, the key limitation remaining was the lack of an efficient and scalable
library for TAGI; as the method does not rely on backprop nor gradient descent,
it is incompatible with traditional libraries such as PyTorch or TensorFlow.
In 2021, Luong-Ha Nguyen decided to lead the development of the new cuTAGI
platform and later on the pyTAGI API with the objective to open the capabilities of TAGI to the entire community.
