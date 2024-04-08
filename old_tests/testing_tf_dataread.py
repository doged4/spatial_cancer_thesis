# See her https://www.tensorflow.org/guide/data
# for background
# %%
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf

# %% Get image patches as tensors

# just for practice

from scipy.io import mmread

genes_33c = mmread("processed_data\Spaceranger_uncompressed\V10F03-033_C\matrix.mtx")
genes_33d = mmread("processed_data\Spaceranger_uncompressed\V10F03-033_D\matrix.mtx")
# %%
# These are tensors of all of gene by barcode
#   number of barcodes are nonuniform
tensor_33c = tf.convert_to_tensor(genes_33c.todense())
tensor_33d = tf.convert_to_tensor(genes_33d.todense())

# together = np.ndarray(genes_33c.shape + (2,))
# together = np.bmat([genes_33c.todense(), genes_33d.todense()])
# %%
