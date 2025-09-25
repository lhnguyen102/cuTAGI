.. _regression_tutorial:

========================================
Regression Tutorial: A Step-by-Step Guide
========================================

This tutorial provides a detailed, step-by-step explanation of the ``examples/regression.py`` script, demonstrating how to use ``pytagi`` for a 1D regression task.

.. note::
   This tutorial requires the toy dataset located in the ``data/toy_example/`` directory. Ensure these files are present before running the example.

Detailed Code Walkthrough
-------------------------

1. Imports and Main Function Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We import necessary libraries, including **NumPy** for array manipulation, **tqdm** for progress bars, and specific components from ``pytagi`` for metrics, data loading, normalization, and network construction.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 1-10

2. Dataset Setup and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script defines the file paths for the 1D toy dataset and initializes the data loaders. The **Training Data Loader** computes mean/std statistics, which are then explicitly passed to the **Test Data Loader** for consistent normalization.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 12-25

3. Network Architecture Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define a simple **three-layer Neural Network** using ``Sequential``: an input layer (1 feature to 50 neurons), a ReLU activation, and an output layer (50 neurons back to 1 output feature).

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 28, 31-35

4. Bayesian Output Updater and Observation Noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We initialize the ``OutputUpdater``, which handles the core TAGI Bayesian inference step for the last layer's weights. We also define the **observation variance** :math:`\sigma_v^2`—the assumed noise/uncertainty in the target data.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 41-43

5. Training Loop: Setup
~~~~~~~~~~~~~~~~~~~~~~~

The training loop iterates over the specified number of epochs, generating data batches and initializing a progress bar for monitoring.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 47-50

6. Inner Training Loop: Forward Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside the batch iteration, we perform the **forward pass** using the network to obtain the predicted mean (:math:`m_{pred}`). Note that the variance (:math:`\_`) is ignored during training.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 53-54

7. Inner Training Loop: Bayesian Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the central step of TAGI. The ``OutputUpdater`` adjusts the posterior distribution (mean and variance) of the final layer's weights based on the observed target data (:math:`y`).

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 56-62

8. Inner Training Loop: Backward Pass and Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the output layer update, we perform the **backward pass** (`net.backward()`) to calculate the necessary delta values for all hidden layers, followed by the **step** (`net.step()`) which updates the weights and variances of the hidden layers.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 64-66

9. Inner Training Loop: Metric Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We calculate the training MSE metric for monitoring purposes. This involves **unstandardizing** the normalized predictions and observations before computing the MSE.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 67-73

10. Testing Setup and Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the training loop, we iterate over the test dataset. In the test phase, we capture both the predicted **mean** (:math:`m_{pred}`) and **variance** (:math:`v_{pred}`) from the network's forward pass.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 77-90

11. Total Uncertainty and Unstandardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We calculate the **total uncertainty** by adding the observation variance (:math:`\sigma_v^2`) back to the predicted network variance (`v_pred + \sigma_v**2`). All collected test data (predictions and inputs) are then unstandardized.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 92-106

12. Final Metrics and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final steps involve calculating the test **MSE** and **Log-likelihood** (a measure of probabilistic model fit), and using the ``PredictionViz`` class to plot the results, including the predicted uncertainty bounds.

.. literalinclude:: ../../../examples/regression.py
    :language: python
    :lines: 108-129
