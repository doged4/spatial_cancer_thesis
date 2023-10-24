### Load images and spot-assosiated data
import os
import numpy as np
import pandas as pd
import csv
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "2500000000000000000"
import cv2
import multiprocess

# Specify the path of config file
CONFIG_FILE_PATH = ".\image_inputters\main_config.csv"

# Read config file with mapping of SpaceRanger output folders to full-resolution images
config = pd.read_csv(CONFIG_FILE_PATH)

# Define reading-sample strategy
def read_sample(sample):

    # Read the full-res tissue image as np.ndarray in two formats
    fu_tissue_image_color = cv2.imread(config.loc[sample,"fullres_path"], cv2.IMREAD_COLOR)
    fu_tissue_image_gray = cv2.cvtColor(fu_tissue_image_color, cv2.COLOR_BGR2GRAY)

    # Get scaling factor for transition of cords from full-res to hi-res and spot diamater
    scale_file = pd.read_json(os.path.join(config.loc[sample,"spaceranger_path"], "outs", "spatial", "scalefactors_json.json"), typ='series')
    spot_diameter = scale_file[0]
    scale_factor = scale_file[1]

    # Get the high-res tissue image as np.ndarray in two formats
    hi_tissue_image_color = cv2.imread(os.path.join(config.loc[sample,"spaceranger_path"], "outs", "spatial", "tissue_hires_image.png"), cv2.IMREAD_COLOR)
    hi_tissue_image_gray = cv2.cvtColor(hi_tissue_image_color, cv2.COLOR_BGR2GRAY)

    # Read spot positions on a tissue
    spot_positions = []
    spot_path = os.path.join(config.loc[sample,"spaceranger_path"], "outs", "spatial", "tissue_positions_list.csv")
    with open(spot_path, newline="") as file: 
        for row in csv.reader(file):
            spot_positions.append({
                "sample_id": str(config.loc[sample,"sample_id"]),
                "patient_id": str(config.loc[sample,"patient"]),
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
    with open(config.loc[sample,"true_annotation_path"], newline="") as file: 
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
    sample_data = {"sample_id": config.loc[sample,"sample_id"],
                    "patient_id": config.loc[sample,"patient"],
                    "fu_tissue_image_gray": fu_tissue_image_gray,
                    "fu_tissue_image_color": fu_tissue_image_color,
                    "hi_tissue_image_gray": hi_tissue_image_gray,
                    "hi_tissue_image_color": hi_tissue_image_color,
                    "spot_diameter_fures": float(spot_diameter),
                    "scale_factor": scale_factor,
                    "spot_data": spot_positions,
                    "spot_expression": [],
                    "patches_collection": []}

    return sample_data, spot_positions

if __name__ == '__main__':
    # Read all samples using multi-cpu
    pool_obj = multiprocess.Pool()
    ans = pool_obj.map(read_sample, config.index)
    pool_obj.close()

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