pytagi.nn.base_layer
====================

.. py:module:: pytagi.nn.base_layer


Classes
-------

.. autoapisummary::

   pytagi.nn.base_layer.BaseLayer


Module Contents
---------------

.. py:class:: BaseLayer

   Base layer class providing common functionality and properties for neural network layers.
   This class acts as a Python wrapper for the C++ backend, exposing layer attributes
   and methods for managing layer information, device placement, and parameters.


   .. py:method:: to_cuda()

      Moves the layer's parameters and computations to the CUDA device.



   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: get_max_num_states() -> int

      Retrieves the maximum number of states the layer can hold.

      :returns: The maximum number of states.
      :rtype: int



   .. py:property:: input_size
      :type: int


      Gets the input size of the layer.


   .. py:property:: output_size
      :type: int


      Gets the output size of the layer.


   .. py:property:: in_width
      :type: int


      Gets the input width of the layer (for convolutional layers).


   .. py:property:: in_height
      :type: int


      Gets the input height of the layer (for convolutional layers).


   .. py:property:: in_channels
      :type: int


      Gets the input channels of the layer (for convolutional layers).


   .. py:property:: out_width
      :type: int


      Gets the output width of the layer (for convolutional layers).


   .. py:property:: out_height
      :type: int


      Gets the output height of the layer (for convolutional layers).


   .. py:property:: out_channels
      :type: int


      Gets the output channels of the layer (for convolutional layers).


   .. py:property:: bias
      :type: bool


      Gets a boolean indicating whether the layer has a bias term.


   .. py:property:: num_weights
      :type: int


      Gets the total number of weights in the layer.


   .. py:property:: num_biases
      :type: int


      Gets the total number of biases in the layer.


   .. py:property:: mu_w
      :type: numpy.ndarray


      Gets the mean of the weights (mu_w) as a NumPy array.


   .. py:property:: var_w
      :type: numpy.ndarray


      Gets the variance of the weights (var_w) as a NumPy array.


   .. py:property:: mu_b
      :type: numpy.ndarray


      Gets the mean of the biases (mu_b) as a NumPy array.


   .. py:property:: var_b
      :type: numpy.ndarray


      Gets the variance of the biases (var_b) as a NumPy array.


   .. py:property:: delta_mu_w
      :type: numpy.ndarray


      Gets the delta mean of the weights (delta_mu_w) as a NumPy array.


   .. py:property:: delta_var_w
      :type: numpy.ndarray


      Gets the delta variance of the weights (delta_var_w) as a NumPy array.
      The delta corresponds to the amount of change induced by the update step.


   .. py:property:: delta_mu_b
      :type: numpy.ndarray


      Gets the delta mean of the biases (delta_mu_b) as a NumPy array.
      This delta corresponds to the amount of change induced by the update step.


   .. py:property:: delta_var_b
      :type: numpy.ndarray


      Gets the delta variance of the biases (delta_var_b) as a NumPy array.
      This delta corresponds to the amount of change induced by the update step.


   .. py:property:: num_threads
      :type: int


      Gets the number of threads to use for computations.


   .. py:property:: training
      :type: bool


      Gets a boolean indicating whether the layer is in training mode.


   .. py:property:: device
      :type: bool


      Gets a boolean indicating whether the layer is on the GPU ('cuda') or CPU ('cpu').
