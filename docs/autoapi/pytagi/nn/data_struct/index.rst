pytagi.nn.data_struct
=====================

.. py:module:: pytagi.nn.data_struct


Classes
-------

.. autoapisummary::

   pytagi.nn.data_struct.BaseHiddenStates
   pytagi.nn.data_struct.BaseDeltaStates
   pytagi.nn.data_struct.HRCSoftmax


Module Contents
---------------

.. py:class:: BaseHiddenStates(size: Optional[int] = None, block_size: Optional[int] = None)

   Represents the base hidden states, acting as a Python wrapper for the C++ backend.
   This class manages the mean (mu_a), variance (var_a), and Jacobian (jcb) of hidden states.


   .. py:property:: mu_a
      :type: List[float]


      Gets or sets the mean of the hidden states (mu_a).


   .. py:property:: var_a
      :type: List[float]


      Gets or sets the variance of the hidden states (var_a).


   .. py:property:: jcb
      :type: List[float]


      Gets or sets the Jacobian of the hidden states (jcb).


   .. py:property:: size
      :type: int


      Gets the size of the hidden states.


   .. py:property:: block_size
      :type: int


      Gets the block size of the hidden states.


   .. py:property:: actual_size
      :type: int


      Gets the actual size of the hidden states.


   .. py:method:: set_input_x(mu_x: List[float], var_x: List[float], block_size: int)

      Sets the input for the hidden states.

      :param mu_x: The mean of the input x.
      :type mu_x: List[float]
      :param var_x: The variance of the input x.
      :type var_x: List[float]
      :param block_size: The block size for the input.
      :type block_size: int



   .. py:method:: get_name() -> str

      Gets the name of the hidden states type.

      :returns: The name of the hidden states type.
      :rtype: str



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str

      Sets a new size and block size for the hidden states.

      :param new_size: The new size.
      :type new_size: int
      :param new_block_size: The new block size.
      :type new_block_size: int

      :returns: A message indicating the success or failure of the operation.
      :rtype: str



.. py:class:: BaseDeltaStates(size: Optional[int] = None, block_size: Optional[int] = None)

   Represents the base delta states, acting as a Python wrapper for the C++ backend.
   This class manages the change in mean (delta_mu) and change in variance (delta_var)
   induced by the update step.


   .. py:property:: delta_mu
      :type: List[float]


      Gets or sets the change in mean of the delta states (delta_mu).


   .. py:property:: delta_var
      :type: List[float]


      Gets or sets the change in variance of the delta states (delta_var).


   .. py:property:: size
      :type: int


      Gets the size of the delta states.


   .. py:property:: block_size
      :type: int


      Gets the block size of the delta states.


   .. py:property:: actual_size
      :type: int


      Gets the actual size of the delta states.


   .. py:method:: get_name() -> str

      Gets the name of the delta states type.

      :returns: The name of the delta states type.
      :rtype: str



   .. py:method:: reset_zeros() -> None

      Reset all delta_mu and delta_var to zeros.



   .. py:method:: copy_from(source: BaseDeltaStates, num_data: int = -1) -> None

      Copy values of delta_mu and delta_var from another delta states object.

      :param source: The source delta states object to copy from.
      :type source: BaseDeltaStates
      :param num_data: The number of data points to copy. Defaults to -1 (all).
      :type num_data: int



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str

      Sets a new size and block size for the delta states.

      :param new_size: The new size.
      :type new_size: int
      :param new_block_size: The new block size.
      :type new_block_size: int

      :returns: A message indicating the success or failure of the operation.
      :rtype: str



.. py:class:: HRCSoftmax

   Hierarchical softmax wrapper from the CPP backend.


   .. py:property:: obs
      :type: List[float]


      Gets or sets the fictive observation \in [-1, 1].


   .. py:property:: idx
      :type: List[int]


      Gets or sets the indices assigned to each label.


   .. py:property:: num_obs
      :type: int


      Gets or sets the number of indices for each label.


   .. py:property:: len
      :type: int


      Gets or sets the length of an observation (e.g., 10 labels -> len(obs) = 11).
