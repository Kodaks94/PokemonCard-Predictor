import tensorflow as tf
from tensorflow.python.platform import build_info

print("TensorFlow version:", tf.__version__)

# Access the build_info dictionary directly
info = build_info.build_info

print("Is built with CUDA:", info["is_cuda_build"])
print("CUDA version:", info["cuda_version"])
print("cuDNN version:", info["cudnn_version"])

print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())