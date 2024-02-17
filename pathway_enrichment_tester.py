# %%
import scanpy as sc
import anndata as ad

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
one_disease_all_control = sc.concat([s8t2_adata[1], s8c2_adata], merge='same', label='dataset')
sc.tl.rank_genes_groups(one_disease_all_control,
                        groupby='slide_condition',
                        method = 'wilcoxon',
                        groups=['Tumor'], 
                        reference='Normal')
#%%[markdown]
# It seems like it refuses to work for our group as it only contains one sample. This makes a lot of sense, and yet we would still really like to be able to find the most relatively enriched genes for these samples.
# We can see the check in `scanpy\tools\_rank_genes_groups.py` on line `100` that singlets result in divide by zero errors.
# %% ssGSEA test
ssgs_result = gseapy.ssgsea(data=s8t2_adata[1].to_df().T,
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
# %%
