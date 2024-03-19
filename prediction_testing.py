# %% [markdown]
# ## Predictive Model Testing
# Cevi Bainton 3/18/2024
# 
# This is a notebook to do testing on prediction of gene set enrichment features produced by `pathway_enricher.py` from image features produced by `image_extracter.py`.
# This is an exploratory notebook.

# %% 
from sklearn import linear_model
import matplotlib.pyplot as plt # In case simple plots needed

from anndata import read_h5ad
import pandas as pd



# %% [markdown]
# We can see that multiple regression is pretty straightforward. 
# %%
fake_lasso_model = linear_model.Lasso(alpha=0) # Fake because alpha = 0 means it acts like ordinary least squares!
fake_lasso_model.fit(X = [[1,0], [2,5]], y = [3,4])
print(fake_lasso_model.coef_)
print(fake_lasso_model.intercept_)
# %%[markdown]
# We can also do multivariate multiple regression.
# We can see that multiple regression is pretty straightforward. 
# %%
mmfake_lasso_model = linear_model.Lasso(alpha=0) # Fake because alpha = 0 means it acts like ordinary least squares!
mmfake_lasso_model.fit(X = [[1,0], [2,5]], y = [[3, 6], [4,3]])
print(mmfake_lasso_model.coef_)
print(mmfake_lasso_model.intercept_)
# %% Elastic net combines lasso and ridge regression
elnet_model = linear_model.ElasticNet(alpha=0.5) # Fake because alpha = 0 means it acts like ordinary least squares!
elnet_model.fit(X = [[1,0], [2,5]], y = [[3, 6], [4,3]])
print(elnet_model.coef_)
print(elnet_model.intercept_)
# %% [markdown]
# ## Multi-Task Elastic Net
# We would really like to fully remove image features we don't need the same way for each gene set we predict. We can use Elastic net for the first part, and we'd like it to work on a lot of output variables at once too (multivariate).
# For that, we need Multi-Task Elastic Net.
# %% [markdown]
# Lets load in the data we have so far. 

s8t2_adata_with_imfeatures = read_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")

# Source adatas differ due to discrepancy in how one was saved, but this will be fixed 
predictors = s8t2_adata_with_imfeatures.obsm['im_features']
s8t2_adata = read_h5ad("./intermediate_data/s8t2_all_at_once_enrichments.h5ad")
response = pd.DataFrame(
    data = s8t2_adata.layers['nes'],
    index = s8t2_adata.obs_names,
    columns = s8t2_adata.var_names
)
small_predictors = predictors.iloc[:2,:]
small_response = response.iloc[:2, :]
# %%[markdown]  Multi task Elastic
# Our model seems to work ok.
mt_elnet = linear_model.MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5)
mt_elnet.fit(X = small_predictors, y=small_response)
# %%
