# %%
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np


# %%
# print(os.environ["CV_IO_MAX_IMAGE_PIXELS"])

# %% [markdown]
# This script is intended as a basic test of methods to split and input images for the model creation.

# %%
high_res_path = "original_data\High-resolution_tissue_images"
patient_tester = "33"
patient_tester_first = R'C:\Users\cbainton\Desktop\ST_project\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.A1-Spot000001.jpg'

h = 12
# Fails to import as too large 
first_full_size = cv2.imread(patient_tester_first)
cv2.imshow("fist test", first_full_size[h *10000:h*10000 + 10000, h*10000:h*10000+ 10000, 0:3])
cv2.waitKey(0)
cv2.destroyAllWindows()


# Imports and shows fine 
first_lbshot = cv2.imread(R'C:\Users\cbainton\Desktop\ST_project\original_data\LB-screenshots\33A.png')
cv2.imshow("First LB screen shot", first_lbshot)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
