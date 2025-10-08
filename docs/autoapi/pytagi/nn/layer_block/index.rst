pytagi.nn.layer_block
=====================

.. py:module:: pytagi.nn.layer_block


Classes
-------

.. autoapisummary::

   pytagi.nn.layer_block.LayerBlock


Module Contents
---------------

.. py:class:: LayerBlock(*layers: pytagi.nn.base_layer.BaseLayer)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A stack of different layers derived from BaseLayer

   Initialize the Sequential model with the given layers.
   :param layers: A variable number of layers (instances of BaseLayer or derived classes).


   .. py:method:: switch_to_cuda()

      Convert all layers to cuda layer



   .. py:property:: layers
      :type: None


      Get layers
