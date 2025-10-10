py/cuTAGI's documentation
====================================

What is py/cuTAGI?
------------------------------------

pyTAGI is a Python frontend for cuTAGI, its high-performance C++/CUDA backend,
implementing Tractable Approximate Gaussian Inference (TAGI) for deep neural
networks. TAGI treats all network parameters and hidden units as Gaussian random
variables and derives closed-form expressions for prior/posterior means,
variances, and covariances, enabling analytic Bayesian learning without relying
on gradient descent or backpropagation. A high-level overview about the TAGI
theory can be found in the :doc:`theory section <api/theory/index>`.

Getting started
---------------

cuTAGI is available on PyPI. To install, execute the following command in Terminal:

.. code-block:: bash

   pip install pytagi

Full installation instructions can be found in the :doc:`installation guide <api/installations>`.

Here is an example for training a classifer using pytagi on MNIST dataset

.. code-block:: python

   from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential
   from pytagi import Utils, HRCSoftmaxMetric
   from examples.data_loader import MnistDataloader

   batch_size = 32
   dtl = MnistDataLoader()
   metric = HRCSoftmaxMetric(num_classes=10)

   net = Sequential(
      Linear(784, 128),
      ReLU(),
      Linear(128, 128),
      ReLU(),
      Linear(128, 11),
   )
   #net.to_device("cuda")

   udt = OutputUpdater(net.device)
   var_y = np.full((batch_size * 4,), 1.0, dtype=np.float32)

   batch_iter = dtl.create_data_loader(batch_size)

   for i, (x, y, idx, label) in enumerate(batch_iter):
      m_pred, v_pred = net(x)
      # Update output layer based on targets
      udt.update_using_indices(net.output_z_buffer, y, var_y, idx, net.input_delta_z_buffer)
      net.backward()
      net.step()
      error_rate = metric.error_rate(m_pred, v_pred, label)
      print(f"Iteration: {i} error rate: {error_rate}")


Visit the :doc:`tutorials <api/tutorials/index>` page to learn how to run differnet models for different datasets.

Features
--------

Some key features of cuTAGI include:

- **Performance-Oriented Kernels**: All kernels of DNN layers are written in C++/CUDA from the scratch, with the utilization of pybind11 for seamless Python integration. It allows running on CPU and CUDA devices through Python API.
- **Broad Architecture Support**: It currently supports the basic layer of DNNs including Linear, CNNs, Transposed CNNs, LSTM, Average Pooling,  normalization, enabling the building of mainstream architectures such as Autoencoders, Transformers, Diffusion Models, and GANs.
- **Model Building and Execution**: Currently, it supports sequential model building, with plans to introduce Eager Execution in the future for better debugging.
- **Multi-GPU Training**: Currently, it supports Distributed Data Parallel (DDP) for multi-GPU setups via NCCL and MPI. Find how to use it in the :doc:`Distributed Data Parallel <api/multi_gpu>` page.
- **Open Platform**: cuTAGI provides open access to its entire codebase. This transparency and accessibility allows researchers and developers to dive deep into the cuTAGI's core functionalities.

cuTAGI targets machine learning researchers and developers, aiming to improve the reliability of neural network outcomes, learning efficiency, and adaptability to different dataset sizes. The Python API, inspired by the PyTorch framework, is designed to quickly onboard researchers for idea exploration.


Contributors
------------

The principal developer of pytagi is Luong Ha Nguyen with the support of James-A. Goulet. The main contributors of the library are:

- Luong Ha Nguyen (main developer)
- James-A. Goulet (theory and supervision)
- Van Dai Vuong (LSTM)
- Miquel Florensa Montilla (Heteroscedasticity and Docker)
- David Wardan (SLSTM)

Acknowledgements
----------------

We acknowledge the financial support from the Natural Sciences and Engineering Research Council of Canada, Hydro-Québec, and the Quebec transportation Ministry and in the development of the pyTAGI library.

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
   api/pytagi_api
   api/tutorials/index
   api/dev_guides
   api/multi_gpu
   api/theory/index
   api/about
