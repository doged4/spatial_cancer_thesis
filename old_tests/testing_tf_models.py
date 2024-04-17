# %%
import os
import itertools

# import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import cv2
import pickle


# See here for basis of tutorial : https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

# %% Version info

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\intermediate_data\patched_data"
# image_path = os.path.join(PATCHED_PATH, R"V10F03-033_C\patches\V10F03-033_C_AAACAAGTATCTCCCA-1_50_102.jpg")
# image = cv2.imread(image_path)
# images = np.expand_dims(image, axis = 0) # makes 3 way tensor to 4 way (makes into 1 element long list of images)

# # %% 
# ### Effnet_b4 below

# %%
# image_predictions = model.predict(images)
# %%



### Effnet v2l as paper recommended
# %% Load in data
# IMAGE_SIZE = (480, 480, 3)
IMAGE_SIZE = (380, 380, 3)

# Load in data from jpg files to make TF Dataset objects
# PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\intermediate_data\patched_data_480"
PATCHED_PATH = R"C:\Users\cbainton\Desktop\ST_project\intermediate_data\patched_data"
OUT_DIR = "im_features"
# OUT_DATASET = "tf_dataset\processed_st_480"
OUT_DATASET = "tf_dataset\processed_st_380_compressed"



# %% Sample setup
# sample = "V10F03-033_C"
samples = os.listdir(PATCHED_PATH)

def save_dataset(sample):
    """Saves image patches in sample as a TensorFlow Dataset as shards
    Args: sample, name of sample to read in; string
    Output: returns nothing, constructs folder sample in OUT_DATASET and populates"""
    # Setup input and output paths
    sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
    image_paths = os.listdir(sample_dir)
    data_out = os.path.join(OUT_DATASET, sample)
    print(f"Sample: {sample}")

    # Create sample output folder if needed
    if not os.path.exists(data_out):
        os.mkdir(data_out)

    # Read in images to 4 way tensor (num patches, image height, image width, 3 colors)
    images = np.ndarray((len(image_paths),) + IMAGE_SIZE)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image_arr = cv2.imread(os.path.join(sample_dir, image_path))
        images[i] = image_arr

    # Get names without .jpg suffix
    dataset_names = tf.constant([name[:-4] for name in image_paths])
    # Construct tf Dataset from tensor, each image with spot names
    dataset = tf.data.Dataset.from_tensor_slices((images, dataset_names))
    # Save to shard format
    dataset.save(data_out, compression = 'GZIP')
    
    print(f"Successfully output to {data_out}")


def get_features(sample, from_dataset = True):
    print(sample)
    print(f'peak {tf.config.experimental.get_memory_info("GPU:0")["peak"]/1000000} mb')
    print(f'current {tf.config.experimental.get_memory_info("GPU:0")["current"]/1000000} mb')

    # sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
    image_paths = os.listdir(sample_dir)
    sample_out = os.path.join(OUT_DIR, sample)

    if not os.path.exists(sample_out):
        os.mkdir(sample_out)
    
    tf.data.Dataset.load() # MAKE LOAD DATASETS

    test =tf.data.Dataset.from_tensor_slices(images)


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



# %% Alternate TF dataset pulling method

def get_image(path, name):
    image = tf.image.decode_jpeg(tf.io.read_file(path))
    return image, name

def pointer_dataset(sample):
    # Setup input and output paths
    sample_dir = os.path.join(PATCHED_PATH, sample, "patches")
    image_paths = os.listdir(sample_dir)
    data_out = os.path.join(OUT_DATASET, sample)
    print(f"Sample: {sample}")
    dataset_names = tf.constant([name[:-4] for name in image_paths])
    # See here: https://stackoverflow.com/questions/44416764/loading-folders-of-images-in-tensorflow
    image_path_tensor = tf.convert_to_tensor(image_paths)
    dataset_names_tensor = tf.convert_to_tensor(dataset_names)
    dataset = tf.data.Dataset.from_tensor_slices((image_path_tensor, dataset_names_tensor))
    dataset.map(get_image, num_parallel_calls=8) # convert paths to image tensors
    print(f"Retrieved {data_out}")

    return dataset
    # Save to shard format
    # dataset.save(data_out, compression = 'GZIP')


# test= pointer_dataset('V10J20-085_D')
# print(test)

# %% Keras image input approach
# seems like best method below:
sample = 'V10F03-033_A'
sample_dir = os.path.join(PATCHED_PATH, sample, r'patches')


# Seems to be 2566 images
image_dataset = tf.keras.utils.image_dataset_from_directory(
    sample_dir,
    image_size = IMAGE_SIZE[0:2],
    labels=None # keeps 
) 

# %% Convert to features
#  Load in efficientnet_b4, which is the right size: EASY TO CHANGW
model_handle = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
# See more info https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
IMAGE_SIZE = (380, 380, 3) # image.shape

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = IMAGE_SIZE),
    hub.KerasLayer(model_handle) # if doesn't work, clear cached folder where weights stored
])
# Note the lack of a dropout layer or a dense layer
model.build((None,)+IMAGE_SIZE)

image_predictions = model.predict(image_dataset, use_multiprocessing=True)

feature_dict = {}
filepaths = os.listdir(sample_dir)

# %% Load image predictions into DF

for i in range(len(filepaths)):
    feature_dict[filepaths[i]] = image_predictions[i]
pd.DataFrame()
# feature_df = pd.DataFrame(feature_dict, columns=['m'+str(x) for x in range()])
# TODO: make into df and save df 

# %% Convert all samples to tf Dataset format and output in OUTPUT_DATASET
# save_dataset('V10J20-085_D')
# done_samples = os.listdir(OUT_DATASET)
# these_samples = filter(lambda x: not x in done_samples, samples)
# for t_sample in these_samples:
#     save_dataset(t_sample)


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

# %% Run samples to get features


# %% Testing out tf model into a model pipeline

#  Load in efficientnet_b4, which is the right size: EASY TO CHANGW
model_handle = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
# See more info https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
IMAGE_SIZE = (380, 380, 3) # image.shape

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = IMAGE_SIZE),
    hub.KerasLayer(model_handle) # if doesn't work, clear cached folder where weights stored
])
# model.build((None,)+IMAGE_SIZE)
model.build(IMAGE_SIZE)

# %% Testing layers
zeroes = tf.constant(0, dtype = tf.int32, shape = IMAGE_SIZE ) # make a zeroes image
ed_zeroes = tf.expand_dims(zeroes, axis=0) # Extra dimension on zeroes
model(ed_zeroes) # Works!

test_layer = hub.KerasLayer(model_handle, trainable=False, dtype = tf.int32, input_shape = IMAGE_SIZE)

# %% Playground with strings
embedding_test = "https://tfhub.dev/google/nnlm-en-dim50/2"
test_strings = hub.KerasLayer(embedding_test, input_shape = [], dtype = tf.string, trainable=True)
small_str = tf.constant(np.array([b'test']),  dtype=tf.string)
test_strings(small_str)
