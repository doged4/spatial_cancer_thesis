# %%
import scanpy as sc
import anndata as ad
import pandas as pd

import matplotlib.pyplot as plt

import gseapy

# %%
s8t2_adata = ad.read_h5ad("./intermediate_data/33D_S8T2.h5ad")
s8c2_adata = ad.read_h5ad("./intermediate_data/33B_S8C2.h5ad")

# %% [markdown]
# Let's do a short filtering step just to start out. This is _not_ our long run filtering choice, but just enough to speed up the later steps.

# %%
# f_disease = sc.pp.norm
# f_control = sc.pp.recipe_seurat(s8c2_adata, plot=True)

# %%
big_adata = ad.concat([s8c2_adata, s8t2_adata], merge='same', label='dataset')
dataset_class = big_adata.obs['slide_condition'].to_list()

# %%
# Advised from GSEAPY read the docs gsea function
gs_out = gseapy.GSEA(data=big_adata.to_df().transpose(),
                 gene_sets='MSigDB_Oncogenic_Signatures',
                 classes=dataset_class,
                 permutation_type='phenotype',
                 permutation_num=1000, # reduce number to speed up test
                 outdir=None,  # do not write output to disk
                 method='signal_to_noise',
                 threads=4, seed= 7
                 )
gs_out.pheno_pos = 'Tumor'
gs_out.pheno_neg = 'Normal'
gs_out.run()
# Fails on no genes if not transposed
# fails on heatmap if transposed?

#%%[markdown]
# We can see our results below. They seem to have calculated the enrichment of each term across all the spots (so likely all tumor vs all control). We'd love this for _one_ tumor vs all control.

# %% Results here
gs_out.res2d.head(15)

#%%[markdown]
# Seems to be plotting the normalized expression of the most important genes to the model specific term of interest. For our purposes, the left columns are those of `S8T2` while the right columns are those of `S8C2`.

# %%
# Taken from scRNA-seq example
## Heatmap of gene expression
i = 7
genes = gs_out.res2d.Lead_genes.iloc[i].split(";") # Get's genes relevant to the term
ax = gseapy.heatmap(df = gs_out.heatmat.loc[genes], # Gene values
           z_score=None, 
           title=gs_out.res2d.Term.iloc[i], # Term name
           figsize=(6,5),
           cmap=plt.cm.viridis,
           xticklabels=False)


# %% Trying DEG to enrichr analysis method as described in paper
# Is this gene rank between groups, or if not how are the genes grouped?
sc.tl.rank_genes_groups(big_adata,
                        groupby='slide_condition',
                        method = 'wilcoxon',
                        groups=['Tumor'], 
                        reference='Normal')

sc.pl.rank_genes_groups(big_adata, n_genes=25, sharey=False)

#%%[markdown]
# Can we do this gene ranking with just one spot?

#%% Trying DEG between one disease spot and all controls
# Fails!
# one_disease_all_control = sc.concat([s8t2_adata[1], s8c2_adata], merge='same', label='dataset')
# sc.tl.rank_genes_groups(one_disease_all_control,
#                         groupby='slide_condition',
#                         method = 'wilcoxon',
#                         groups=['Tumor'], 
#                         reference='Normal')
#%%[markdown]
# It seems like it refuses to work for our group as it only contains one sample. This makes a lot of sense, and yet we would still really like to be able to find the most relatively enriched genes for these samples.
# We can see the check in `scanpy\tools\_rank_genes_groups.py` on line `100` that singlets result in divide by zero errors.
# %% ssGSEA test
ssgs_result = gseapy.ssgsea(data=s8t2_adata[0:2].to_df().T,
               gene_sets='MSigDB_Oncogenic_Signatures',
               )
#%%[markdown]
# ssGSEA seems to run on just the one sample. What does it tell us?
# %%
ssgs_result.res2d

# %%[markdown]
# It gives us an enrichment score, and a normalized enrichement score! This is really interesting. 
# 
# TODO: Next questions to address:
# * Normalize to control tissue prior to ssGSEA
# * How can we what does NES mean relative to ES? Does either have a relative probability?
#

#%%[markdown]
# Let's compare running a single sample through ssGSEA vs multiple
# %% Examine the independence of running ssgsea
spot_ssgs_results = {}
spot_ssgs_results[s8t2_adata.obs.index[0]] = gseapy.ssgsea(
    data=s8t2_adata[0].to_df().T,
    gene_sets='MSigDB_Oncogenic_Signatures',
    ).res2d

spot_ssgs_results[s8t2_adata.obs.index[1]] = gseapy.ssgsea(
    data=s8t2_adata[1].to_df().T,
    gene_sets='MSigDB_Oncogenic_Signatures',
    ).res2d

all_at_once  = gseapy.ssgsea(
    data=s8t2_adata[0:2].to_df().T,
    gene_sets='MSigDB_Oncogenic_Signatures',
    ).res2d


# %% Check similarity
print(spot_ssgs_results[s8t2_adata.obs.index[0]] ==  all_at_once.loc[
    all_at_once['Name'].str.match(s8t2_adata.obs.index[0]),:].reset_index(drop=True))

print(spot_ssgs_results[s8t2_adata.obs.index[1]] ==  all_at_once.loc[
    all_at_once['Name'].str.match(s8t2_adata.obs.index[1]),:].reset_index(drop=True))

#%%[markdown]
# It seems like the enrichment socres are identical, but the normalization differs
# TODO: investigave the changes in normalization

#%%[markdonw]
# Let's try running them all through ssGSEA solo. We can then collect all of the enrichment into an adata at the end
# %% Run all spots
for i, spot_id in enumerate(s8t2_adata.obs.index):
    spot_ssgs_results[spot_id] = gseapy.ssgsea(
        data=s8t2_adata[i].to_df().T,
        gene_sets='MSigDB_Oncogenic_Signatures',
        )
# %%
# Collects res2d from each ssgsea output in dictionary
res_dict = {}
norm_res_dict = {}
for key in spot_ssgs_results.keys():
    res_dict[key] = spot_ssgs_results[key].res2d['ES']
    norm_res_dict[key] = spot_ssgs_results[key].res2d['NES']

# %% [markdown]
# We now want to transpose our results to fit the expected format of an AnnData object that ScanPy likes. We will also get the column names from the term names, and cast to float to avoid errors when we later set up object.
# %% 
# Get pathway feature names
terms = spot_ssgs_results[list(spot_ssgs_results.keys())[0]].res2d['Term']

# Collects the columns of the res2d for larger df
enriched_df = pd.DataFrame(res_dict).T
enriched_df.columns = list(terms)
enriched_df = enriched_df.astype('float')


# Collects the normalized enrichment for larger df as well ** 
norm_enriched_df = pd.DataFrame(norm_res_dict).T
norm_enriched_df.columns = list(terms)
norm_enriched_df = norm_enriched_df.astype('float')

#%% Genes
# msig = gseapy.Msigdb()
# # Shoul be same as "MSigDB_Oncogenic_Signatures"
# c6_dict = msig.get_gmt(category='c6.all')

# %% [markdown]
#  We can keep all of the spot related and unstructured data from our previous adata. Then all we do is set new vars and our new results as out `X`.
# %% Run all spots
# Collect all the outputs into one larger adata

t_results_adata = ad.AnnData(
    X = enriched_df,
    obs=s8t2_adata.obs,
    var = pd.DataFrame(data = {'term' : list(terms)}, index = list(terms)),
    uns=s8t2_adata.uns,
    obsm=s8t2_adata.obsm,
    layers={'normalized_per_spot' : norm_enriched_df}
)

# Just for safekeeping!
t_results_adata.write_h5ad("intermediate_data/s8t2_one_by_one_enrichments.h5ad")

# %% [markdown]
# Look at this plot! Scanpy works nicely for us. `SRC UP.V1 UP` looks nice for us
# %% Plot pathway features spatially.
# Works!
sc.pl.spatial(t_results_adata, img_key="hires", color = "SRC UP.V1 UP")
# %%
# Return different dtypes hmm
s8t2_adata[1,2].to_df().loc[:,'OR4F5']

t_results_adata[1,2].to_df().loc[:,"RB P107 DN.V1 UP"]


#%% [markdown]
# ## All at once
# We will now try to run the GSEAPY ssGSEA on all of the samples at once rather than through a for loop, as this may improve performance.

#%% Run all together
all_ssgs_results = gseapy.ssgsea(
        data=s8t2_adata.to_df().T,
        gene_sets='MSigDB_Oncogenic_Signatures',
        )
#%% Convert to anndata

