# Developement ground for the image extraction pipeline
# relies on image_extracter.py

# %% Load in libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
import numpy as np



from image_extracter import image_extracter

# %% Load in data
s8t2_adata = ad.read_h5ad("./intermediate_data/33D_S8T2.h5ad")

# %% Set up image extracter
# extracter = image_extracter()
# extracter.prep_model()
#%% Testing
# import cv2

# test_tile = cv2.imread(R"intermediate_data/patched_data/V10F03-033_A/patches/V10F03-033_A_AAACACCAATAACTGC-1_59_19.jpg")
# extracter.extract_one(test_tile)

# # %% Set up image container with image
# im_container = sq.im.ImageContainer()
# im_container = im_container.from_adata(s8t2_adata)
# TODO: This initializes with the hires image
# We want an image container full of each spot image

# %%
# big_im_container = sq.im.ImageContainer()
# big_im_container.add_img(R".\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.D1-Spot000001.jpg")
# # Seems to use the adata.uns and the spatial coords in adata to get image?
# generator = big_im_container.generate_spot_crops(s8t2_adata, 
#                                                  return_obs=True)
# # Gets tuple of spot image and image container
# spot_image = generator.__next__()
# # Generates to size 377 here, based on math for the slide

# # Reshape out an extra dimension
# plt.imshow(spot_image[0]['image_0'].to_numpy().reshape((377,377,3)))
# spot_image[0].show() # Also works

# new_extracter = image_extracter(image_size= (377,377,3))
# new_extracter.prep_model()

# # Works!
# new_extracter.extract_one(
#     spot_image[0]['image_0']
# )
# # %% Test if underlying functions for feature extraction work
# spot_image[0].features_custom(func = new_extracter.extract_one, layer = 'image_0')
# %%[markdown]
# This shows that we gnerate the features just fine, but as a single entry for each spot which is a single tensor
# %% 
# Does this work?
# sq.im.calculate_image_features(
#     s8t2_adata, 
#     big_im_container,
#     features= 'summary'
# )

# %% 
# Untested
# The wrapper should set adata.obsm
# Doesn't seem to work yet
# Crashes repeatedly
# sq.im.calculate_image_features(
#     adata = s8t2_adata,
#     img = spot_image[0],
#     features = 'custom',
#     key_added = 'features',
#     # n_jobs = 4,
#     features_kwargs={"custom": {"func":  new_extracter.extract_one}})

# TODO:
#   Either
#       make image container with pretiled images, see what attaches them to previous location
#       or make image container filetype work with image_extracter class
#           seems to have extra dimension?
#   Look at this guide for help? https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html
# %%[markdown]
# ## For loop method
# %% Attempt to manually run through and set obsm
# Initialize the output array
# s8t2_adata.obsm['im_features'] = np.empty(shape = (s8t2_adata.shape[0], new_extracter.num_features), 
# 
from time import process_time
import logging

logging.basicConfig(filename="log_image_extraction_tester.log", format='%(asctime)s - %(message)s', level=logging.DEBUG)

logging.debug(f"Started: {process_time()}\n")

big_im_container = sq.im.ImageContainer()
big_im_container.add_img(R".\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.D1-Spot000001.jpg")
# Seems to use the adata.uns and the spatial coords in adata to get image?
generator = big_im_container.generate_spot_crops(s8t2_adata, 
                                                 return_obs=True)


new_extracter = image_extracter(image_size= (377,377,3))
new_extracter.prep_model()

# Initialize image feature array as empty
# Renaming the columns here is necessary to avoid un readable adata
empty_array = np.full(fill_value= np.nan,
                        shape = (s8t2_adata.shape[0], new_extracter.num_features), 
                        dtype=np.float32)
# Set empty array as .obsm['im_features']
s8t2_adata.obsm['im_features'] = pd.DataFrame(empty_array, 
                                                index = s8t2_adata.obs_names,
                                                columns = [f"feature_{x}" for x in range(new_extracter.num_features)])

for i, (image_container, spot_index) in enumerate(generator):
    s8t2_adata.obsm['im_features'].loc[spot_index,:] = new_extracter.extract_one(
        image_container['image_0'], 
        as_np=True)
    if i % 10 == 0:
        # print(f"Spot number:{i} at time {process_time()}")
        logging.debug(f"Spot number {i}")


# Seems to fail if adata does not have correct column names in .obsm['im_features']
s8t2_adata.write_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")
s8t2_adata.obsm['im_features'].to_csv('./intermediate_data/im_features_s8t2.csv')

logging.debug(f"End: {process_time()}\n")

# s8t2_adata_with_imfeatures = ad.read_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")
# %%[markdown]
# ## Tensorflow training syntax method. See here [https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html] for inspiration.
# %%
# # put some of below into image extracter
# import tensorflow as tf

# second_generator = big_im_container.generate_spot_crops(s8t2_adata, 
#                                                  return_obs=True)
# dataset = tf.data.Dataset.from_tensor_slices ([x for x in second_generator])
# # tf.data.Dataset.from_generator # seems to be not parallelizable according to those online
# spot_names = tf.data.Dataset.from_tensor_slices(s8t2_adata.obs_names) # takes a while but may end up faster?


# # make tf dataset from slice
#   using generate_image_crops
# zip with labels (here just names?)
# embedding = model.predict [still use image_extracter]
# construct new anndata from embeddings
# Then merge in old anndata columns with new?
