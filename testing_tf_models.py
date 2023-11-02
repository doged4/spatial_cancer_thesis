import os
import itertools

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub

# See here for basis of tutorial : https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

# %% Version info

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# %%
# Test different image analysis models
# Just on patches
# Export with just image features