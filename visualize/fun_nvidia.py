def fun_nvidia():
  """
  Check if we are using a Nvidia gpu.

  Returns:
    and string with GPU name if Nvidia GPU is available, otherwise None.
  """
    # Check for GPU
    try:
        import tensorflow as tf
        gpu_name = tf.test.gpu_device_name()
        if gpu_name:
            return gpu_name
    except ImportError:
        return None
    return None