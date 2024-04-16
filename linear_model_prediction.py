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
import glob

# %% Data params
IMAGE_FEATURES_DIR = "intermediate_data/batch_extracted_image_adatas"
ENRICHMENTS_DIR = "intermediate_data/enrichments_on_updn_de"

MODEL_OUT_DIR = "intermediate_data/linear_models"
# %% Get our data
def get_data_as_dfs(im_features_dir, enrichments_dir, nonnormalized = False, exclude_ids = []):
    enr_dfs = []
    im_dfs = []
    spot_info_dfs = []

    missed_spots = []
    # id = "S20T1"
    biopsy_ids = [x.split('\\')[-1].split("_")[0] for x in glob.glob(enrichments_dir + "/*h5ad")]
    for id in biopsy_ids:
        if id in exclude_ids:
            continue
        enr_ad = read_h5ad(f"{enrichments_dir}/{id}_de_gene_enrichments.h5ad")
        im_ad = read_h5ad(f"{im_features_dir}/{id}_im.h5ad")

        if nonnormalized:
            enr_df = pd.DataFrame(enr_ad.layers['nonnormalized'], index = enr_ad.obs_names, columns = enr_ad.var_names)
        else:
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
def get_n_splits(n, X, spot_info):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=n)
    return list(
        splitter.split(X, spot_info.loc[:,['classification', 'biopsy_sample_id']]))
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
        n_alphas=20,
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
        sys.stdout = StreamToLogger(log, logging.INFO, background=False)
        sys.stderr = StreamToLogger(log, logging.ERROR, background=False)

        logging.basicConfig(filename= "model_training.log", 
                            format='%(asctime)s - %(message)s', 
                            level=logging.DEBUG, filemode='a')
        logging.debug("Starting CV Elnet run")
        
        logging.debug("Get data")
        im_features, enrichments, spot_info = get_data_as_dfs(im_features_dir=IMAGE_FEATURES_DIR, enrichments_dir=ENRICHMENTS_DIR)
        # im_features.to_csv(R"intermediate_data\temp\im_features_temp.csv")
        # enrichments.to_csv(R"intermediate_data\temp\enrichments_temp.csv")
        # spot_info.to_csv(R"intermediate_data\temp\spot_info_temp.csv")
        # im_features = pd.read_csv("im_features_temp.csv")
        # enrichments = pd.read_csv("enrichments_temp.csv")
        # spot_info = pd.read_csv("spot_info_temp.csv")
        logging.debug("Data retrieved")
        model_splits = get_n_splits(5, im_features, spot_info)
        logging.debug("Model running")
        cv_elnet = model2(im_features, enrichments, stratify=model_splits)
        logging.debug("Model run, saving")
        pickle.dump(cv_elnet, open(MODEL_OUT_DIR + "/proper_cv_elnet.pkl", "wb"))

        logging.debug(f"Run completed")