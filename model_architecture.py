# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from anndata import read_h5ad
from sklearn.model_selection import train_test_split
# from pandas import DataFrame
import image_extracter
from linear_model_prediction import get_data_as_dfs, get_n_splits
from pandas import DataFrame

# Inspired somewhat by this page: https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html

# %% Parameters
IMAGE_SHAPE = (380, 380, 3)
PRETRAINED_HANDLE = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b4-feature-vector/1"
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

INPUT_IMAGES_DIR = "intermediate_data/patched_data/" 
# BIOPSY_ID_ADDED = '_S8T2'

# TODO: this currently uses just C6 enrichments
# INPUT_ENRICHMENTS_PATH = "intermediate_data/s8t2_all_at_once_enrichments.h5ad"

DUMMY_IMAGE_FEATURES_DIR = "intermediate_data/batch_extracted_image_adatas"
ENRICHMENTS_DIR = "intermediate_data/enrichments_on_updn_de"

TEST_NAME = 'first full test'

RANDOM_STATE = 12
USE_SAVED = True
# %% Load in enrichments
# all_enrichments = read_h5ad(INPUT_ENRICHMENTS_PATH).to_df()
# spot_names = list(all_enrichments.index) 
# We assume that all spots in enrichments are what we care about

# %%
# Get enrichments and class info
_, enrichments, spot_info = get_data_as_dfs(im_features_dir=DUMMY_IMAGE_FEATURES_DIR, enrichments_dir=ENRICHMENTS_DIR)
spot_names = list(enrichments.index)
# Train test split spots
if not USE_SAVED: 

    train_spots, test_spots = train_test_split(
        spot_names, 
        test_size = 0.3,
        train_size = 0.7,
        random_state = RANDOM_STATE,
        stratify=spot_info.loc[:,["classification", "biopsy_sample_id"]]
    )
    # Get suffixes for generation
    train_spot_suffixes = ["_" + x.split("_")[-1] for x in train_spots]
    test_spot_suffixes = ["_" + x.split("_")[-1] for x in test_spots]

if not USE_SAVED:
    helper_extracter = image_extracter.image_extracter()

def dataset_from_spot_names(spot_names, image_dir, enrichments_df, name_append = ""):
    """From list of spot names retrieve image data and enrichments as tf.Datasets
        Arguments:
            spot_names : list of names
            image_dir  : dir to search for spots in by barcode
            enrichments_df: dataframe of enrichment scores to predict
            name_append : string to append to the name of every image name, used internally
        Returns:
            tf.Dataset of zipped (image, enrichments)"""
    # barcodes = [image_dir + "*" + x.split('_')[0] + "*" for x in spot_names]
    folder_w_barcode = [f"{image_dir}*{x.split('_')[-2][:-1]}_{x.split('_')[-2][-1]}*/patches/*{x.split('_')[0]}*" 
                        for x in spot_names]

    image_set = helper_extracter.image_set_from_path(folder_w_barcode, 
                                                in_place = False, 
                                                name_append = name_append) # until spot naming is repaired
    images = image_set['images']


    enrichments = tf.data.Dataset.from_tensor_slices(enrichments_df.loc[spot_names, :])

    return tf.data.Dataset.zip((images, enrichments))
    

# %% Create datasets
# Training set
if not USE_SAVED:
    train_set = dataset_from_spot_names(train_spots, INPUT_IMAGES_DIR, enrichments, train_spot_suffixes)

    # Test set
    test_set = dataset_from_spot_names(test_spots, INPUT_IMAGES_DIR, enrichments, test_spot_suffixes)

    train_set.save(f"intermediate_data/tf_dataset_train_test/saved_train_rs_{RANDOM_STATE}")
    test_set.save(f"intermediate_data/tf_dataset_train_test/saved_test_rs_{RANDOM_STATE}")
else:
    train_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_train_rs_{RANDOM_STATE}")
    test_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_test_rs_{RANDOM_STATE}")

# %%
# Model Setup
# Not actually set here
ACTIVATION_FUNC = 'relu'
LEARNING_RATE = 1e-4
EPOCHS = 10

# - Model layers
input_layer = tf.keras.Input(IMAGE_SHAPE)
pretrained_layer = hub.KerasLayer(PRETRAINED_HANDLE, trainable=True)
dense_out_layer = tf.keras.layers.Dense(
    units=enrichments.shape[1],
    activation=ACTIVATION_FUNC
)
# - Model architecture
model = tf.keras.Sequential([
    input_layer,
    pretrained_layer,
    dense_out_layer
])

# - Compile Model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
              loss = tf.keras.losses.MeanSquaredError()
              )
# %%
def run_fit():
    # Fit model
    model_loss = model.fit(
        x = train_set,
        validation_data = test_set,
        epochs = EPOCHS,
        verbose = 2 # seems helpful?
    )

    # Save model
    model.save(f"models/{TEST_NAME}.keras")
    DataFrame(model_loss.history).to_csv(f"models/{TEST_NAME}_history.csv")
# %% 
# Run and log
if __name__ == '__main__':
    import logging
    import sys
    import os

    # Look here for guidance: https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file

    # file_handler = logging.FileHandler("transfer_learning.log")
    # stdout_handler = logging.StreamHandler(sys.stdout)
    # Use StreamToLogger?
    logging.basicConfig(filename = "transfer_learning.log", format='%(asctime)s - %(message)s',
                        level=logging.DEBUG, filemode = 'a')
    logging.debug("Beginning training")
    param_dict = {
            'pid':os.getpid(),
            'IMAGE_SHAPE' : IMAGE_SHAPE,
            'PRETRAINED_HANDLE' : PRETRAINED_HANDLE,
            'INPUT_IMAGES_DIR' : INPUT_IMAGES_DIR,
            'USE_SAVED [takes precedent over other data loading]' : USE_SAVED,
            'RANDOM_TT_SPLIT_STATE': RANDOM_STATE,
            'ACTIVATION_FUNC' : ACTIVATION_FUNC,
            'LEARNING_RATE' : LEARNING_RATE,
            'EPOCHS' : EPOCHS,
            'TEST_NAME' : TEST_NAME
        }
    logging.debug(f"Params are:{param_dict}")
    run_fit()

    logging.debug("Traing complete")

# %%
