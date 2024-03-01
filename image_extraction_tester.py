# Developement ground for the image extraction pipeline
# relies on image_extracter.py

# %% Load in libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import squidpy as sq


from image_extracter import image_extracter

# %% Load in data
s8t2_adata = ad.read_h5ad("./intermediate_data/33D_S8T2.h5ad")

# %% Set up image extracter
extracter = image_extracter()
extracter.prep_model()
#%% Testing
import cv2

test_tile = cv2.imread(R"intermediate_data/patched_data/V10F03-033_A/patches/V10F03-033_A_AAACACCAATAACTGC-1_59_19.jpg")
extracter.extract_one(test_tile)

# %% Set up image container with image
im_container = sq.im.ImageContainer()
im_container = im_container.from_adata(s8t2_adata)
# TODO: This initializes with the hires image
# We want an image container full of each spot image