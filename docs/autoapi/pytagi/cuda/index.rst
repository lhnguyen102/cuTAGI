pytagi.cuda
===========

.. py:module:: pytagi.cuda


Functions
---------

.. autoapisummary::

   pytagi.cuda.is_available
   pytagi.cuda.is_nccl_available
   pytagi.cuda.get_device_count
   pytagi.cuda.get_current_device
   pytagi.cuda.set_device
   pytagi.cuda.is_device_available
   pytagi.cuda.get_device_properties
   pytagi.cuda.get_device_memory


Module Contents
---------------

.. py:function:: is_available() -> bool

   Check if CUDA is available

   :returns: True if CUDA is available, False otherwise
   :rtype: bool


.. py:function:: is_nccl_available() -> bool

   Check if NCCL is available

   :returns: True if NCCL is available, False otherwise
   :rtype: bool


.. py:function:: get_device_count() -> int

   Get the number of CUDA devices

   :returns: Number of CUDA devices
   :rtype: int


.. py:function:: get_current_device() -> int

   Get the current CUDA device

   :returns: Current CUDA device
   :rtype: int


.. py:function:: set_device(device_index: int) -> bool

   Set the current CUDA device

   :param device_index: Device index to set


.. py:function:: is_device_available(device_index: int) -> bool

   Check if a specific CUDA device is available

   :param device_index: Device index to check


.. py:function:: get_device_properties(device_index: int) -> str

   Get the properties of a specific CUDA device

   :param device_index: Device index to get properties of


.. py:function:: get_device_memory(device_index: int) -> Tuple[int, int]

   Get the memory of a specific CUDA device

   :param device_index: Device index to get memory of


