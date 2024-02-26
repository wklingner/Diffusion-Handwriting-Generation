import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
  print("You have GPUs available!")
  print("Default GPU device:")
  print(physical_devices[0])
else:
  print("You don't have GPUs available.")
