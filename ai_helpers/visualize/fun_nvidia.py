def fun_nvidia():
  """
  Check if we are using a Nvidia gpu.

  Returns:
    A string with GPU name if Nvidia GPU is available, otherwise None.
  """
  # Check for GPU
  try:
      import tensorflow as tf
      gpu_name = tf.test.gpu_device_name()
      if gpu_name:
          print(f"Nvidia GPU detected: {gpu_name}")  # Print GPU information
          return gpu_name
      else:
          print("No Nvidia GPU detected.")
  except ImportError:
      print("TensorFlow not found. Unable to check for GPU.")
      return None
  return None