pytagi.nn.output_updater
========================

.. py:module:: pytagi.nn.output_updater


Classes
-------

.. autoapisummary::

   pytagi.nn.output_updater.OutputUpdater


Module Contents
---------------

.. py:class:: OutputUpdater(model_device: str)

   .. py:method:: update(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:method:: update_using_indices(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, selected_idx: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:method:: update_heteros(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:property:: device
      :type: str
