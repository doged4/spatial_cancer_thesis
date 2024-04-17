# %% [markdown]
# ## Predictive Model Testing
# Cevi Bainton 3/18/2024
# 
# This is a notebook to do testing on prediction of gene set enrichment features produced by `pathway_enricher.py` from image features produced by `image_extracter.py`.
# This is an exploratory notebook.

# %% 
import sklearn.linear_model
import matplotlib.pyplot as plt # In case simple plots needed
sklearn.set_config(enable_metadata_routing=True)

from anndata import read_h5ad
import pandas as pd
import scanpy as sc

import pickle
import logging
import os
# %% [markdown]
# # Goals of our model
# We want a linear model which is able to use many explanatory features (image embedding) with other random effects (patient and slide) and predict many response variables (enrichment scores).
# We'd like a model that can do:
# * Multiple prediction with regularization (many inputs that it subselects)
# * Multivariate regression (many outputs)
# * Captures random effects (batch effects of patient or slide)

# %% [markdown]
# # We can see that multiple regression is pretty straightforward. 
# # %%
# fake_lasso_model = sklearn.linear_modl.Lasso(alpha=0) # Fake because alpha = 0 means it acts like ordinary least squares!
# fake_lasso_model.fit(X = [[1,0], [2,5]], y = [3,4])
# print(fake_lasso_model.coef_)
# print(fake_lasso_model.intercept_)
# # %%[markdown]
# # We can also do multivariate multiple regression.
# # We can see that multiple regression is pretty straightforward. 
# # %%
# mmfake_lasso_model = sklearn.linear_modl.Lasso(alpha=0) # Fake because alpha = 0 means it acts like ordinary least squares!
# mmfake_lasso_model.fit(X = [[1,0], [2,5]], y = [[3, 6], [4,3]])
# print(mmfake_lasso_model.coef_)
# print(mmfake_lasso_model.intercept_)
# # %% Elastic net combines lasso and ridge regression
# elnet_model = sklearn.linear_modl.ElasticNet(alpha=0.5) # Fake because alpha = 0 means it acts like ordinary least squares!
# elnet_model.fit(X = [[1,0], [2,5]], y = [[3, 6], [4,3]])
# print(elnet_model.coef_)
# print(elnet_model.intercept_)
# # %% [markdown]
# # ## Multi-Task Elastic Net
# # We would really like to fully remove image features we don't need the same way for each gene set we predict. We can use Elastic net for the first part, and we'd like it to work on a lot of output variables at once too (multivariate).
# # For that, we need Multi-Task Elastic Net.
# # %% [markdown]
# # Lets load in the data we have so far. 

# s8t2_adata_with_imfeatures = read_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")

# # Source adatas differ due to discrepancy in how one was saved, but this will be fixed 
# predictors = s8t2_adata_with_imfeatures.obsm['im_features']
# s8t2_adata = read_h5ad("./intermediate_data/s8t2_all_at_once_enrichments.h5ad")
# response = pd.DataFrame(
#     data = s8t2_adata.layers['nes'],
#     index = s8t2_adata.obs_names,
#     columns = s8t2_adata.var_names
# )
# small_predictors = predictors.iloc[:2,:]
# small_response = response.iloc[:2, :]
# # %%[markdown]  Multi task Elastic
# # Our model seems to work ok.
# mt_elnet = sklearn.linear_modl.MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5)
# mt_elnet.fit(X = small_predictors, y=small_response)
# # %%

# %% [markdown]
# ## Model 1: Modelling a single slide with `MultiTask ElasticNet`
# Let's try modelling all of a single slide with just elasticnet
def model1(im_features_ad, enrichments_ad ):

    mt_elnet_model = sklearn.linear_modl.MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5)
    mt_elnet_model.fit(
        X = im_features_ad.to_df(),
        y = enrichments_ad.to_df()
    )
    return mt_elnet_model

# %% Retrieve model

# mt_elnet = pickle.load(open('intermediate_data/saved_elnet.pickle','rb'))

# %% [markdown]
# ### Model 1 results
# Let's see what we have modelled.
# im_features_ad = read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")
# enrichments_ad = read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")

# mt_elnet = model1(im_features_ad=im_features_ad,
#                   enrichments_ad=enrichments_ad)


# r2_val = mt_elnet.score( X = im_features_ad.to_df(),
#     y = enrichments_ad.to_df())
# r2_val
# %%



# %% [markdown]
# ## Model 2: Modelling a single slide with `CV MultiTask ElasticNet`
# Let's try modelling all of a single slide with cross validated elasticnet. When `l1_ratio` is 1, we are doing pure LASSO. When it is 0, we are doing pure ridge regression.

def model2(im_features_ad, enrichments_ad):
    mt_elnet_model = sklearn.linear_modl.MultiTaskElasticNetCV(
        l1_ratio= [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        eps=1e-3,
        n_alphas=100,
        cv=5,
        n_jobs=10
    )
    
    mt_elnet_model.fit(
        X = im_features_ad.to_df(),
        y = enrichments_ad.to_df(),
        enable_metadata_routing=True
    )
    return mt_elnet_model


# # %% Retrieve model

# mt_elnet = pickle.load(open('intermediate_data/linear_models/saved_elnet.pickle','rb'))
# # %% [markdown]
# # ### Looking at our model
# mt_elnet = pickle.load(open('intermediate_data/linear_models/saved_cv_elnet.pickle','rb'))
# # Selected parameters:
# mt_elnet.alpha_
# mt_elnet.l1_ratio_

# mt_elnet.intercept_

# # %% [markdown]
# # Let's look at our coefficients
# mt_elnet.coef_
# # %% [markdown]
# # Seems like a lot of zeroes... how many features aren't included?
# # Every feature that is all zero:
# num_unused = sum([all([x == 0 for x in y]) for y in mt_elnet.coef_.transpose()])
# print(f"{len(mt_elnet.feature_names_in_)-num_unused}/{len(mt_elnet.feature_names_in_)} features used")
# # What is not all zeroes?
# used_features = [not all([x == 0 for x in y]) for y in mt_elnet.coef_.transpose()]
# used_features_names = mt_elnet.feature_names_in_[used_features]
# print(used_features_names)

# # %% [markdown]
# # # Evaluating our model
# im_features_ad = read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")
# enrichments_ad = read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")

# path_to_of_descent = mt_elnet.path(im_features_ad.to_df(), enrichments_ad.to_df())

# # This doesn't seem good
# score = mt_elnet.score(
#     X = im_features_ad.to_df(),
#     y = enrichments_ad.to_df()
# )
# print(f"R^2 is {score}")
# # %% [markdown]
# # This is our R^2 value. It is concerning that it is so low.
# # %%
# from yellowbrick.regressor import ResidualsPlot
# resids = ResidualsPlot(mt_elnet)
# resids.fit(
#     X = im_features_ad.to_df(),
#     y = enrichments_ad.to_df())
# # %% [markdown]
# # That's not good! That's quite concerning. What do these features look like?
# sc.pl.spatial(im_features_ad, 
#               color = used_features_names, 
#               bw=True) # bw for clarity
# # %% [markdown]
# # It looks like these image features are pretty sparse. This could mean a couple of things.
# # I trust the regularization, but let's go back to figure out what the best model would capture.
# linear_output = sklearn.linear_modl.LinearRegression()
# linear_output.fit(
#     X = im_features_ad.to_df(),
#     y = enrichments_ad.to_df())
# # Here's what our best could be:
# print(linear_output.score(
#     X = im_features_ad.to_df(),
#     y = enrichments_ad.to_df()))
# #
# # %% [markdown]
# # Seems like a lot of our features are being eliminated. This could either mean that our regulization is too agressive, or that we just do not have enough nonsparse features.
# # This leaves us with two options:
# # 
# # * Refit the model with different parameters
# * Transfer learning train our model to do enrichment predictions
# %% Backround logging info


if __name__ == '__main__':
    logging.basicConfig(filename="model_training.log", format='%(asctime)s - %(message)s', level=logging.DEBUG)
    logging.debug(f"Process running at: {os.getpid()}")
    logging.debug(f"Running prediction_testing.py")
    logging.debug(f"Started Model CV MT Elasticnet Training\n")

    im_features_ad = read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")
    enrichments_ad = read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")

    mt_elnet = model2(im_features_ad, enrichments_ad)

    logging.debug(f"End\n")
    with open('intermediate_data/saved_cv_elnet_with_meta.pickle','wb') as file:
        pickle.dump(mt_elnet, file=file)
    logging.debug("Saved model to pickle")