# Author: Cevi Bainton
# Note: This pipeline is DEPRECATED (and never worked) in favor of patch_pipeline.py
#      This code is an implementation of a pipeline created by Kacper Maciejewski
#       see https://gitlab.com/daub-lab/image-analysis_spatial-transcriptomics_breast-cancer_summer23
#       And the WSI-ST data handling README for details

# 10/23/23 will load in some images, taking a long time, has not run fully

# %% Libraries
### Load images and spot-assosiated data
import os
import numpy as np
import pandas as pd
import csv
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "2500000000000000000"
import cv2
import multiprocess
from math import ceil

import pickle

# Set to relevant dir on remote
os.chdir("/Users/cbainton/Desktop/ST_project")

# %% From cevis_A-img_loader_anno_multi.py

# Specify the path of config file
CONFIG_FILE_PATH = ".\image_inputters\main_config_altered.csv"

# Read config file with mapping of SpaceRanger output folders to full-resolution images
config = pd.read_csv(CONFIG_FILE_PATH)

# Define reading-sample strategy
def read_sample(sample, local_config):
    # Set 
    # fu_tissue_image_color = cv2.imread(local_config.loc[sample,"fullres_path"], cv2.IMREAD_COLOR)
    # fu_tissue_image_gray = cv2.cvtColor(fu_tissue_image_color, cv2.COLOR_BGR2GRAY)

    # Get scaling factor for transition of cords from full-res to hi-res and spot diamater
    scale_file = pd.read_json(os.path.join(local_config.loc[sample,"spaceranger_path"], "outs", "spatial", "scalefactors_json.json"), typ='series')
    spot_diameter = scale_file.iloc[0]
    scale_factor = scale_file.iloc[1]

    # Get the high-res tissue image as np.ndarray in two formats
    hi_tissue_image_color = cv2.imread(os.path.join(local_config.loc[sample,"spaceranger_path"], "outs", "spatial", "tissue_hires_image.png"), cv2.IMREAD_COLOR)
    hi_tissue_image_gray = cv2.cvtColor(hi_tissue_image_color, cv2.COLOR_BGR2GRAY)

    # Read spot positions on a tissue
    spot_positions = []
    spot_path = os.path.join(local_config.loc[sample,"spaceranger_path"], "outs", "spatial", "tissue_positions_list.csv")
    with open(spot_path, newline="") as file: 
        for row in csv.reader(file):
            spot_positions.append({
                "sample_id": str(local_config.loc[sample,"sample_id"]),
                "patient_id": str(local_config.loc[sample,"patient"]),
                "barcode": str(row[0]),
                "in_tissue": True if row[1] == "1" else False,
                "array_col": int(row[3]), #col == x
                "array_row": int(row[2]), #row == y
                "pxl_col_hires": int(row[5])*scale_factor,
                "pxl_row_hires": int(row[4])*scale_factor,
                "pxl_col_fures": int(row[5]),
                "pxl_row_fures": int(row[4]),
            })
    spot_positions = pd.DataFrame(spot_positions)

    # Include only spots on tissue
    spot_positions = spot_positions[spot_positions["in_tissue"] == True]

    # Read true expert annotations
    true_labels = []
    print(local_config.loc[sample,"true_annotation_path"])
    with open(local_config.loc[sample,"true_annotation_path"], newline="") as file: 
        for row in csv.reader(file):
            true_labels.append({
                "barcode": str(row[0]),
                "st_cluster": int(row[1]),
                "st_label": str(row[2]),
                "clinical_cluster": int(row[3]),
                "clinical_label": str(row[4])
            })
    true_labels = pd.DataFrame(true_labels)
    
    # Match true expert annotations with spot data by barcodes
    # and include only spots for which annotation is available
    spot_positions = spot_positions.merge(true_labels, on="barcode", how="inner")

    # Combine all sample information
    sample_data = {"sample_id": local_config.loc[sample,"sample_id"],
                    "patient_id": local_config.loc[sample,"patient"],
                    # "fu_tissue_image_gray": fu_tissue_image_gray,
                    # "fu_tissue_image_color": fu_tissue_image_color,
                    "hi_tissue_image_gray": hi_tissue_image_gray,
                    "hi_tissue_image_color": hi_tissue_image_color,
                    "spot_diameter_fures": float(spot_diameter),
                    "scale_factor": scale_factor,
                    "spot_data": spot_positions,
                    "spot_expression": [],
                    "patches_collection": []}

    return sample_data, spot_positions

# %% Continuing above into main function

# if __name__ == '__main__':
    # Note that must be run with the above '__main__' check on Windows for multiprocessing

reader_args = [[x, config] for x in config.index]
# Initialize starmpa args
# ans = []

# NUM_POOLS = 6
# group_size =  ceil(len(reader_args) / NUM_POOLS)
# print(f"Samples in {NUM_POOLS} groups")
# print("Starting sample making...")
# for i in range(0,NUM_POOLS):
#     print(f"Starting batch {i}")

#     small_ans = []
#     upper = min(i*group_size + group_size, len(reader_args))
#     small_reader_args = reader_args[i*group_size:upper]

#     pool_obj = multiprocess.Pool()
#     # Read all samples using multi-cpu
#     # multiprocess.Value("local_config", config) # added
#     # ans = pool_obj.map(read_sample, config.index)
#     small_ans = pool_obj.starmap(read_sample, small_reader_args)
#     pool_obj.close()
#     ans = ans + small_ans
#     print(f"Done with batch {i}")

ans = [read_sample(samp_info[0], samp_info[1]) for samp_info in reader_args]

# Using for loop here
# one of the pathe may not exist, check against kacpers code
# ans = [read_sample(sample_index) for sample_index in config.index]

# Create 'samples' and 'total' objects
samples = []
total = {"spots": pd.DataFrame(), "expression": pd.DataFrame(), "patches": []}
for sample_data, spot_positions in ans:
    samples.append(sample_data)
    total["spots"] = pd.concat([total["spots"], spot_positions])

# Convert 'samples' into a dataframe
    samples_df = pd.DataFrame(samples)

# Print recommended patch size based on mean spot diameter rounded up
print(f"Mean spot diameter in full-res: {int(np.ceil(np.mean(samples_df['spot_diameter_fures'])))} (min: {int(np.ceil(min(samples_df['spot_diameter_fures'])))}, max: {int(np.ceil(max(samples_df['spot_diameter_fures'])))})")

# %% From WSI/B_patcher_raw.py
### Create patches/tiles of WSIs for every sample

# Set the patch size
PATCH_SIZE = 380
PATCH_TYPE = "color"

# Enforce combined list is empty
total["patches"] = []

# Iterate over samples
for sample in samples:
    sample["patches_collection"] = []

    # Iterate over spots
    for col, row in zip(sample["spot_data"]["pxl_col_fures"], sample["spot_data"]["pxl_row_fures"]):
        patch_col = int(col - PATCH_SIZE / 2)
        patch_row = int(row - PATCH_SIZE / 2)
        
        # Exctract a patch for every spot
        patch_image = cv2.cvtColor(sample[f"fu_tissue_image_{PATCH_TYPE}"][patch_col:patch_col+PATCH_SIZE, patch_row:patch_row+PATCH_SIZE], cv2.COLOR_BGR2RGB)

        # Make sure a patch to append exists
        if patch_image.size:
            sample["patches_collection"].append(patch_image)

    # Add sample patches to combined list
    total["patches"].extend(sample["patches_collection"])

pickle.dump(total, open(".\image_inputters\inputter_results\total.pickle", "wb"))
samples_df.to_csv(".\image_inputters\inputter_results\samples.csv")

print(type(total))
print(len(samples_df))
# %% From C-exp_loader_norm.ipynb

# The expression normalization is not run here, but will be tested in future