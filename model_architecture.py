# %% Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from anndata import read_h5ad
from sklearn.model_selection import train_test_split
# from pandas import DataFrame
import image_extracter
from linear_model_prediction import get_data_as_dfs, get_n_splits
from pandas import DataFrame
import sys

# Inspired somewhat by this page: https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html

# %% Parameters
IMAGE_SHAPE = (380, 380, 3)
PRETRAINED_HANDLE = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b4-feature-vector/1"
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

INPUT_IMAGES_DIR = "intermediate_data/patched_data/" 


DUMMY_IMAGE_FEATURES_DIR = "intermediate_data/batch_extracted_image_adatas"

if len(sys.argv) == 1:
    TEST_NAME  = 'full_test_nn_8_dropout'
else:
    TEST_NAME = sys.argv[1]
if len(sys.argv) > 2:
    ENRICHMENTS_DIR = sys.argv[2]
else:
    ENRICHMENTS_DIR = "intermediate_data/enrichments_on_updn_de"

is_de = 'de' in ENRICHMENTS_DIR


TT_SPLIT_FRAC = 0.8

RANDOM_STATE = 12
USE_SAVED = True

NONNORMALIZED = True
HOLDOUT_BIOPSIES =["S36T2"] # TODO: fix this sample holdout
# %% Load in enrichments
# all_enrichments = read_h5ad(INPUT_ENRICHMENTS_PATH).to_df()
# spot_names = list(all_enrichments.index) 
# We assume that all spots in enrichments are what we care about
print(f"Start up run: {TEST_NAME}")
# %%
# Get enrichments and class info
_, enrichments, spot_info = get_data_as_dfs(im_features_dir=DUMMY_IMAGE_FEATURES_DIR, enrichments_dir=ENRICHMENTS_DIR,
                                            nonnormalized=NONNORMALIZED, is_de = is_de)
# Remove biopsies that have been designated as holdouts
for biop in HOLDOUT_BIOPSIES:
    enrichments = enrichments.filter(regex=f"^\w*-1_\d\d\D_(?!{biop})", axis=0)
    spot_info = spot_info.filter(regex=f"^\w*-1_\d\d\D_(?!{biop})", axis=0)
# Get spot names for dataset making
spot_names = list(enrichments.index)


# spot_names
# Train test split spots
if not USE_SAVED: 
    if NONNORMALIZED:
        # sd_nn_enrichments = enrichments.std(axis=0)
        # mean_nn_enrichments = enrichments.mean(axis=0)
        # enrichments = enrichments.sub(mean_nn_enrichments, axis=1).div(sd_nn_enrichments, axis = 1)

        # DIVIDE BY MEAN
        # sd_nn_enrichments = enrichments.std(axis=0)
        for slide in spot_info['biopsy_sample_id'].unique():
            temp_means = enrichments.loc[spot_info['biopsy_sample_id'] == slide, :].mean(axis=0)
            enrichments.loc[spot_info['biopsy_sample_id'] == slide, :] = enrichments.loc[
                spot_info['biopsy_sample_id'] == slide, :].div(temp_means, axis=1)

    train_spots, test_spots = train_test_split(
        spot_names, 
        test_size = 1-TT_SPLIT_FRAC,
        train_size = TT_SPLIT_FRAC,
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

    train_set.save(f"intermediate_data/tf_dataset_train_test/saved_train_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_norm_{NONNORMALIZED}_hold_{'-'.join(HOLDOUT_BIOPSIES)}")
    test_set.save(f"intermediate_data/tf_dataset_train_test/saved_test_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_norm_{NONNORMALIZED}_hold_{'-'.join(HOLDOUT_BIOPSIES)}")
else:
    # train_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_train_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_nnorm_{NONNORMALIZED}")
    # test_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_test_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_nnorm_{NONNORMALIZED}")
    train_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_train_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_norm_{NONNORMALIZED}_hold_{'-'.join(HOLDOUT_BIOPSIES)}")
    test_set = tf.data.Dataset.load(f"intermediate_data/tf_dataset_train_test/saved_test_rs_{RANDOM_STATE}_ttsplit_{TT_SPLIT_FRAC}_norm_{NONNORMALIZED}_hold_{'-'.join(HOLDOUT_BIOPSIES)}")

# %%
# Model Setup
# Not actually set here
ACTIVATION_FUNC = 'relu'
LEARNING_RATE = 1e-4
EPOCHS = 20
LOAD_MODEL = False
DROPOUT = 0.5

# - Model layers
input_layer = tf.keras.Input(IMAGE_SHAPE)
pretrained_layer = hub.KerasLayer(PRETRAINED_HANDLE, trainable=True)
dropout_layer = tf.keras.layers.Dropout(rate = DROPOUT)
dense_out_layer = tf.keras.layers.Dense(
    units=enrichments.shape[1],
    activation=ACTIVATION_FUNC
)
# - Model architecture
model = tf.keras.Sequential([
    input_layer,
    pretrained_layer,
    dropout_layer,
    dense_out_layer
])
# poop
# - Compile Model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
              loss = tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()]
              )
checkpoint_filepath = f"./intermediate_data/ckpts/{TEST_NAME}/checkpoint.weights.h5"
backup_dir = f"./intermediate_data/ckpts/{TEST_NAME}/backup"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

model_backup_callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir, save_freq="epoch", delete_checkpoint=False
)

model_history_callback = tf.keras.callbacks.CSVLogger(f"models/{TEST_NAME}_running_history.csv")

# %%
def run_fit():
    # Fit model
    model_loss = model.fit(
        x = train_set,
        validation_data = test_set,
        epochs = EPOCHS,
        verbose = 1, # seems helpful?
        callbacks=[model_checkpoint_callback, model_backup_callback, model_history_callback]
    )

    # Save model
    # TODO: CHANGE NAME BELOW
    model.save(f"models/{TEST_NAME}_nn_{NONNORMALIZED}.keras")
    DataFrame(model_loss.history).to_csv(f"models/{TEST_NAME}_history.csv")
# def model_from_check_point():
#     model = tf.keras.models.load_model(checkpoint_filepath)
#     # Get from checkpoint
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
    print("Beginning training")
    param_dict = {
            'pid':os.getpid(),
            'IMAGE_SHAPE' : IMAGE_SHAPE,
            'PRETRAINED_HANDLE' : PRETRAINED_HANDLE,
            'INPUT_IMAGES_DIR' : INPUT_IMAGES_DIR,
            'USE_SAVED [takes precedent over other data loading]' : USE_SAVED,
            'RANDOM_TT_SPLIT_STATE': RANDOM_STATE,
            'ACTIVATION_FUNC' : ACTIVATION_FUNC,
            'LEARNING_RATE' : LEARNING_RATE,
            'DROPOUT' : DROPOUT,
            'LOAD_MODEL': LOAD_MODEL,
            'EPOCHS' : EPOCHS,
            'TEST_NAME' : TEST_NAME
        }
    if not os.path.exists(f"./intermediate_data/ckpts/{TEST_NAME}/"):
        os.mkdir(f"./intermediate_data/ckpts/{TEST_NAME}/")
    
    logging.debug(f"Params are:{param_dict}")
    print(param_dict)
    if LOAD_MODEL:
        model.load_weights(checkpoint_filepath)
    run_fit()

    logging.debug("Traing complete")
    print("Training complete")

# %% 
