# %%
import os
import itertools

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import cv2
import pickle


# See here for basis of tutorial : https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

# %% Version info

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")



# ### Effnet_b4 below
# # %% Load in efficientnet_b4, which is the right size: EASY TO CHANGW
# model_handle = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"


# # Test different image analysis models
# # Just on patches
# # Export with just image features
# # %%

# PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\patched_data"
# image_path = os.path.join(PATCHED_PATH, R"V10F03-033_C\patches\V10F03-033_C_AAACAAGTATCTCCCA-1_50_102.jpg")
# image = cv2.imread(image_path)
# images = np.expand_dims(image, axis = 0) # makes 3 way tensor to 4 way (makes into 1 element long list of images)

# # %% 
# # See architecture
# # effnet.summary()
# # %% See more info https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
# IMAGE_SIZE = (380, 380, 3) # image.shape

# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape = IMAGE_SIZE),
#     hub.KerasLayer(model_handle)
# ])
# # Note the lack of a dropout layer or a dense layer
# model.build((None,)+IMAGE_SIZE)
# # %%
# image_predictions = model.predict(images)
# # %%



### Effnet v2l as paper recommended
# %% Load in data
IMAGE_SIZE = (480, 480, 3)
PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\patched_data_480"
OUT_DIR = "im_features"

# %% Set up model
effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)
# %% Sample setup
sample = "V10F03-033_C"
sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
image_paths = os.listdir(sample_dir)
sample_out = os.path.join(OUT_DIR, sample)

if not os.path.exists(sample_out):
    os.mkdir(sample_out)

# %% Read in images
images = np.ndarray((len(image_paths),) + IMAGE_SIZE)
for i in range(len(image_paths)):
    image_path = image_paths[i]
    image_arr = cv2.imread(os.path.join(sample_dir, image_path))
    images[i] = image_arr

# plt.imshow(images[0].astype(int))
# plt.show()

# %% Run predictions
image_outputs = effnet.predict(images)

# %%
# Improve this output method in future
#   improve spot naming
#   improve format?
image_out_dict = {}
for i in range(len(image_paths)):
    image_out_dict[image_paths[i]] = image_outputs[i]

out_dict =  os.path.join(sample_out, "image_features_dict.pickle")
pickle.dump(image_out_dict, open(out_dict, "wb"))
# %%
