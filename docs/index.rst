py/cuTAGI's documentation
====================

What is py/cuTAGI?
---------------

py/cuTAGI is an open-source Bayesian neural networks library that is based on the Tractable Approximate Gaussian Inference (TAGI) theory.
It supports various neural network architectures such as fully-connected, convolutional, and transpose convolutional layers,
as well as skip connections, pooling and normalization layers. cuTAGI is capable of performing different tasks such as supervised,
unsupervised, and reinforcement learning. This library has a python API called pyTAGI that allows users to easily use the C++ and CUDA libraries.

How does it work?
-----------------

...

Getting started
---------------

To get started with using our library, check out our:

- installation guide for Windows, MacOS, and Linux (CPU + GPU).
- quick tutorial for a 1D toy problem.

Contributors
------------

....

Acknowledgements
----------------

We acknowledge the financial support from ...

License
-------

py/cuTAGI is released under the MIT license.

**THIS IS AN OPEN SOURCE SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. NO WARRANTY EXPRESSED OR IMPLIED.**

Related references
------------------

**Papers**


- `Tractable approximate Gaussian inference for Bayesian neural networks <https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf>`_ (James-A. Goulet, Luong-Ha Nguyen, and Said Amiri. *JMLR*, 2021)
- `Analytically tractable hidden-states inference in Bayesian neural networks <https://www.jmlr.org/papers/volume23/21-0758/21-0758.pdf>`_ (Luong-Ha Nguyen and James-A. Goulet. *JMLR*, 2022)
- `Analytically tractable inference in deep neural networks <https://arxiv.org/pdf/2103.05461>`_ (Luong-Ha Nguyen and James-A. Goulet. *ArXiv*, 2021)
- `Analytically tractable Bayesian deep Q-Learning <https://arxiv.org/pdf/2106.11086>`_ (Luong-Ha Nguyen and James-A. Goulet. *ArXiv*, 2021)
- `Coupling LSTM Neural Networks and State-Space Models through Analytically Tractable Inference <https://www.sciencedirect.com/science/article/pii/S0169207024000335>`_ (Van Dai Vuong, Luong-Ha Nguyen and James-A. Goulet. *International Journal of Forecasting*, 2024)


.. toctree::
   :maxdepth: 2
   :hidden:

   api/installations
   api/theory
   api/examples
   api/pytagi_api
   api/dev_guides
