# %%
import os
import itertools

# import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import cv2
import pickle


# See here for basis of tutorial : https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

# %% Version info

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# sess = tf.Session()
# run_metadata = tf.RunMetadata()
# sess.run(c, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)

# with open("im_features/run2.txt", "w") as out:
#   out.write(str(run_metadata))

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

# Load in data from jpg files to make TF Dataset objects
PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\patched_data_480"
OUT_DIR = "im_features"
OUT_DATASET = "tf_dataset\processed_st"



# %% Sample setup
# sample = "V10F03-033_C"
samples = os.listdir(PATCHED_PATH)

def save_dataset(sample):
    sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
    image_paths = os.listdir(sample_dir)
    data_out = os.path.join(OUT_DATASET, sample)
    print(f"Sample: {sample}")

    if not os.path.exists(data_out):
        os.mkdir(data_out)

    # Read in images
    images = np.ndarray((len(image_paths),) + IMAGE_SIZE)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image_arr = cv2.imread(os.path.join(sample_dir, image_path))
        images[i] = image_arr

    dataset_names = tf.constant([name[:-4] for name in image_paths])
    dataset = tf.data.Dataset.from_tensor_slices((images, dataset_names))
    dataset.save(data_out)
    print(f"Successfully output to {data_out}")


def get_features(sample):
    print(sample)
    print(f'peak {tf.config.experimental.get_memory_info("GPU:0")["peak"]/1000000} mb')
    print(f'current {tf.config.experimental.get_memory_info("GPU:0")["current"]/1000000} mb')

    # sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
    image_paths = os.listdir(sample_dir)
    sample_out = os.path.join(OUT_DIR, sample)

    if not os.path.exists(sample_out):
        os.mkdir(sample_out)

    # Read in images
    images = np.ndarray((len(image_paths),) + IMAGE_SIZE)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image_arr = cv2.imread(os.path.join(sample_dir, image_path))
        images[i] = image_arr

    test =tf.data.Dataset.from_tensor_slices(images)
    # plt.imshow(images[0].astype(int))
    # plt.show()

    # Run predictions
    image_outputs = effnet.predict(images)

    # 
    # Improve this output method in future
    #   improve spot naming
    #   improve format?
    image_out_dict = {}
    for i in range(len(image_paths)):
        image_out_dict[image_paths[i]] = image_outputs[i]

    out_dict =  os.path.join(sample_out, "image_features_dict.pickle")
    pickle.dump(image_out_dict, open(out_dict, "wb"))

# # %% Set up model
# effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax',
#     include_preprocessing=True
# )
# %%
for t_sample in samples:
    save_dataset(t_sample)
# %%
