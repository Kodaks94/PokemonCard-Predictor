import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU devices found:", tf.config.list_physical_devices('GPU'))

if tf.test.is_built_with_cuda():
    print("TensorFlow was built with CUDA support.")
else:
    print("TensorFlow was NOT built with CUDA support.")
