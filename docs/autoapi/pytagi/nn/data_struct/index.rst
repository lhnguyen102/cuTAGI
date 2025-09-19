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

   .. py:property:: mu_a
      :type: List[float]



   .. py:property:: var_a
      :type: List[float]



   .. py:property:: jcb
      :type: List[float]



   .. py:property:: size
      :type: int



   .. py:property:: block_size
      :type: int



   .. py:property:: actual_size
      :type: int



   .. py:method:: set_input_x(mu_x: List[float], var_x: List[float], block_size: int)


   .. py:method:: get_name() -> str


   .. py:method:: set_size(new_size: int, new_block_size: int) -> str


.. py:class:: BaseDeltaStates(size: Optional[int] = None, block_size: Optional[int] = None)

   .. py:property:: delta_mu
      :type: List[float]



   .. py:property:: delta_var
      :type: List[float]



   .. py:property:: size
      :type: int



   .. py:property:: block_size
      :type: int



   .. py:property:: actual_size
      :type: int



   .. py:method:: get_name() -> str


   .. py:method:: reset_zeros() -> None

      Reset all delta_mu and delta_var to zeros



   .. py:method:: copy_from(source: BaseDeltaStates, num_data: int = -1) -> None

      Copy values of delta_mu and delta_var from delta states



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str


.. py:class:: HRCSoftmax

   Hierarchical softmax wrapper from the CPP backend. Further details can be
   found here https://building-babylon.net/2017/08/01/hierarchical-softmax

   .. attribute:: obs

      A fictive observation \in [-1, 1]

   .. attribute:: idx

      Indices assigned to each label

   .. attribute:: num_obs

      Number of indices for each label

   .. attribute:: len

      Length of an observation e.g 10 labels -> len(obs) = 11


   .. py:property:: obs
      :type: List[float]



   .. py:property:: idx
      :type: List[int]



   .. py:property:: num_obs
      :type: int



   .. py:property:: len
      :type: int
