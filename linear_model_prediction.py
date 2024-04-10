#%% [markdown]
# This model is a simplified fork of ideas learned in `prediction_testing.py`
# %% Libraries
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn import model_selection
sklearn.set_config(enable_metadata_routing=True)

from log_tools.StreamToLogger import StreamToLogger
import logging
import sys
import os
# import matplotlib.pyplot as plt # In case simple plots needed
import pandas as pd
from anndata import read_h5ad
import pickle

# %% Data params
IMAGE_FEATURES_DIR = "intermediate_data/batch_extracted_image_adatas"
ENRICHMENTS_DIR = "intermediate_data/enrichments_on_updn_de"

MODEL_OUT_DIR = "intermediate_data/linear_models"
# %% Get our data
def get_data_as_dfs():
    enr_dfs = []
    im_dfs = []
    spot_info_dfs = []

    missed_spots = []
    # id = "S20T1"
    biopsy_ids = [x.split("_")[0] for x in os.listdir(ENRICHMENTS_DIR)]
    for id in biopsy_ids:
        enr_ad = read_h5ad(f"{ENRICHMENTS_DIR}/{id}_de_gene_enrichments.h5ad")
        im_ad = read_h5ad(f"{IMAGE_FEATURES_DIR}/{id}_im.h5ad")

        enr_df = enr_ad.to_df()
        # Only the columns we care about
        enr_spot_info = enr_ad.obs.iloc[:,:7].copy()

        im_df = im_ad.to_df()
        spots = [x for x in enr_df.index if x in im_df.index]
        spots = pd.Index(spots)
        
        missed_spots += [x for x in enr_df.index if not x in im_df.index]

        im_df = im_df.loc[spots,  :]
        enr_df = enr_df.loc[spots, :]
        enr_spot_info = enr_spot_info.loc[spots, :]


        enr_dfs += [enr_df.copy()]
        im_dfs += [im_df.copy()]
        spot_info_dfs += [enr_spot_info.copy()]

    enrichments = pd.concat(enr_dfs, axis=0)
    im_features = pd.concat(im_dfs, axis=0)
    spot_info = pd.concat(spot_info_dfs, axis =0)

    assert all([x[0] == x[1] for x in zip(enrichments.index, im_features.index)])
    return im_features, enrichments, spot_info

# No idea how these happened for 1390 spots
# print(missed_spots)
# %% Train test split
def get_n_splits(n):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=n)
    return list(
        splitter.split(im_features, spot_info.loc[:,['classification', 'biopsy_sample_id']]))
# %% Linear model with cross validation
# base_model = sklearn.linear_model.LinearRegression()
# base_model.fit(
#     X = im_features,
#     y = enrichments
# )
# logging.debug(f"R^2 is {base_model.score(im_features, enrichments)}")
# print(f"R^2 is {base_model.score(im_features, enrichments)}")


def model2(X, y, stratify = None):
    if stratify == None:
        cv = 5
    else:
        cv = stratify
    mt_elnet_model = sklearn.linear_model.MultiTaskElasticNetCV(
        l1_ratio= [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        eps=1e-3,
        n_alphas=100,
        cv=cv,
        n_jobs=10
    )
    
    mt_elnet_model.fit(
        X = X,
        y = y
    )
    return mt_elnet_model
# %%
DO_LOGGING = True
if __name__ == '__main__':
    
    if DO_LOGGING:
        log = logging.getLogger('log_all')
        sys.stdout = StreamToLogger(log, logging.INFO)
        sys.stderr = StreamToLogger(log, logging.ERROR)

        logging.basicConfig(filename= "model_training.log", 
                            format='%(asctime)s - %(message)s', 
                            level=logging.DEBUG, filemode='a')
        logging.debug("Starting CV Elnet run")
        
        logging.debug("Get data")
        im_features, enrichments, spot_info = get_data_as_dfs()
        logging.debug("Data retrieved")
        model_splits = get_n_splits(5)
        logging.debug("Model running")
        cv_elnet = model2(im_features, enrichments, stratify=model_splits)
        logging.debug("Model run, saving")
        pickle.dump(cv_elnet, open(MODEL_OUT_DIR + "/proper_cv_elnet.pkl", "w"))

        logging.debug(f"Run completed")