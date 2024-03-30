# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from anndata import read_h5ad
from sklearn.model_selection import train_test_split
# from pandas import DataFrame
import image_extracter

# Inspired somewhat by this page: https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html

# %% Parameters
IMAGE_SHAPE = (380, 380, 3)
PRETRAINED_HANDLE = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

INPUT_IMAGE_PATH = "intermediate_data/patched_data/V10F03-033_D/patches/"
BIOPSY_ID_ADDED = '_S8T2'

# TODO: this currently uses just C6 enrichments
INPUT_ENRICHMENTS_PATH = "intermediate_data/s8t2_all_at_once_enrichments.h5ad"

# %% Load in enrichments
all_enrichments = read_h5ad(INPUT_ENRICHMENTS_PATH).to_df()
spot_names = list(all_enrichments.index) 
# We assume that all spots in enrichments are what we care about

# %%
# Train test split spots
train_spots, test_spots = train_test_split(
    spot_names, 
    test_size = 0.3,
    train_size = 0.7,
    random_state = 12,
    # TODO: stratify by classification

)

# %% 
helper_extracter = image_extracter.image_extracter()

def dataset_from_spot_names(spot_names, image_dir, enrichments_adata, name_append = ""):
    """From list of spot names retrieve image data and enrichments as tf.Datasets
        Arguments:
            spot_names : list of names
            image_dir  : dir to search for spots in by barcode
            enrichments_adata: adata of enrichment scores to predict
            name_append : string to append to the name of every image name, used internally
        Returns:
            tf.Dataset of zipped (image, enrichments)"""
    barcodes = [image_dir + "*" + x.split('_')[0] + "*" for x in spot_names]
    image_set = helper_extracter.image_set_from_path(barcodes, 
                                                in_place = False, 
                                                name_append = name_append) # until spot naming is repaired
    images = image_set['images']


    enrichments = tf.data.Dataset.from_tensor_slices(enrichments_adata.loc[spot_names, :])

    return tf.data.Dataset.zip((images, enrichments))
    

# Training set
train_set = dataset_from_spot_names(train_spots, INPUT_IMAGE_PATH, all_enrichments, BIOPSY_ID_ADDED)

# Test set
test_set = dataset_from_spot_names(test_spots, INPUT_IMAGE_PATH, all_enrichments, BIOPSY_ID_ADDED)


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
    units=all_enrichments.shape[1],
    activation='relu'
)
# - Model architecture
model = tf.keras.Sequential([
    input_layer,
    pretrained_layer,
    dense_out_layer
])

# - Compile Model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss = tf.keras.losses.MeanSquaredError()
              )
# %%
model_loss = model.fit(
    x = train_set,
    validation_data = test_set,
    epochs = 10,
    verbose = 2 # seems helpful?
)
# %% 
# Save model
TEST_NAME = 'test1'
model.save('models/{TEST_NAME}.keras')
# %% 
# Run and log
import logging
import sys

# Look here for guidance: https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file

file_handler = logging.FileHandler("transfer_learning.log")
stdout_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(handlers=[file_handler, stdout_handler], format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.debug("Beginning training")
logging.debug(f"Params are:{
    {
        'IMAGE_SHAPE' : IMAGE_SHAPE,
        'PRETRAINED_HANDLE' : PRETRAINED_HANDLE,
        'INPUT_IMAGE_PATH' : INPUT_IMAGE_PATH,
        'BIOPSY_ID_ADDED' : BIOPSY_ID_ADDED,
        'ACTIVATION_FUNC' : ACTIVATION_FUNC,
        'LEARNING_RATE' : LEARNING_RATE,
        'EPOCHS' : EPOCHS
    }
}")

logging.debug("Done training")
