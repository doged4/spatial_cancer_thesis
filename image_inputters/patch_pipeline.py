# Author: Cevi Bainton
# Note: this code is inspired by a pipeline created by Kacper Maciejewski
#       see https://gitlab.com/daub-lab/image-analysis_spatial-transcriptomics_breast-cancer_summer23
#       And the WSI-ST data handling README for details



# sample
#   |-- full res
#   |-- hi res
#   |-- df of meta data
#   |-- df of scale factors
#   |-- patches
#        |-- SAMPLE_BARCODE_COORDS.jpg
#        ...


# %% Libraries
### Load images and spot-assosiated data
import os
import numpy as np
import pandas as pd
import csv
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "2500000000000000000"
import cv2

import pickle

os.chdir("/Users/cbainton/Desktop/ST_project")

CONFIG_FILE_PATH = ".\image_inputters\main_config_altered.csv"


config = pd.read_csv(CONFIG_FILE_PATH)

# %% Define helper functions
def read_sample(sample):
    """
    Read in row from config data frame, return df of relevant meta data for patient
    Args: sample, a row from the config file; dictionary with patient, sample_id, spaceranger_path, fullres_path, true_annotation_path attributes
    """
    # Get image paths in original dataq
    full_res_path = sample["fullres_path"] # Reference
    hi_res_path = os.path.join(sample["spaceranger_path"], "outs", "spatial", "tissue_hires_image.png") # Reference
    
    # Read in scaling factor from full to hi res
    scale_file = pd.read_json(os.path.join(sample["spaceranger_path"], "outs", "spatial", "scalefactors_json.json"), typ='series')
    spot_diameter = scale_file.iloc[0]
    scale_factor = scale_file.iloc[1]


    # Read spot positions on a tissue
    spot_positions = []
    spot_path = os.path.join(sample["spaceranger_path"], "outs", "spatial", "tissue_positions_list.csv")
    with open(spot_path, newline="") as file: 
        for row in csv.reader(file):
            spot_positions.append({
                "sample_id": str(sample["sample_id"]),
                "patient_id": str(sample["patient"]),
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
    
    # Read true expert annotations
    true_labels = []
    print(sample["true_annotation_path"])
    with open(sample["true_annotation_path"], newline="") as file: 
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
    sample_data = {"sample_id": sample["sample_id"],
                    "patient_id": sample["patient"],
                    "fu_image_color_path": full_res_path,
                    "hi_image_color_path": hi_res_path,
                    "spot_diameter_fures": float(spot_diameter),
                    "scale_factor": scale_factor,
                    "spot_data": spot_positions,
                    "spot_expression": [],
                    "patches_collection": []}
    
    return sample_data, spot_positions

PATCH_SIZE = 380

def get_patches (fu_res_path, spot_positions, out_prefix):
    """
    Take full resolution images and slice into patches of PATCH_SIZE.
    Args:   fu_res_path, path to the full resolution image; string
            spot_positions, spot position output by read_sample; pd.DataFrame
            out_prefix, path to output all patches to; string
            *PATCH_SIZE, a constant set to specify height and width; int
    """
    full_image = cv2.imread(fu_res_path)

    for spot_ind in spot_positions.index:
        spot = spot_positions.loc[spot_ind]
        
        sample_id = spot["sample_id"]
        barcode = spot["barcode"]
        array_row = spot["array_row"]
        array_col = spot["array_col"]

        row_coord = spot["pxl_row_fures"] - PATCH_SIZE // 2
        col_coord = spot["pxl_col_fures"] - PATCH_SIZE // 2
        
        patch_name = f"{sample_id}_{barcode}_{array_row}_{array_col}.jpg"
        patch = full_image[row_coord:row_coord + PATCH_SIZE, col_coord:col_coord + PATCH_SIZE]

        cv2.imwrite(os.path.join(out_prefix, patch_name), patch)



# %%
def construct_sample_dir(sample, out_dir):
    sample_folder = os.path.join(out_dir,sample["sample_id"])
    os.mkdir(sample_folder)
    print(sample["sample_id"])
    print("Reading sample info")
    sample_data, spot_positions = read_sample(sample)
    print("Info read")
    spot_positions.to_csv(os.path.join(sample_folder, "spot_positions.csv"))
    pickle.dump(sample_data, open(os.path.join(sample_folder, "sample_data_dict.pickle"), "wb"))

    patches_folder = os.path.join(sample_folder, "patches")
    os.mkdir(patches_folder)
    print("Making patches")
    get_patches(sample_data["fu_image_color_path"], spot_positions, patches_folder)

# %% Main loop over all samples
for sample_ind in config.index:
    construct_sample_dir(config.loc[sample_ind], "patched_data")
# %%
