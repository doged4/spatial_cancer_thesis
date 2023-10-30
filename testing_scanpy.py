import scanpy as sc

# %% Test with h5

this_path = "original_data/Spaceranger_analysis/V10F03-033_A/outs/filtered_feature_bc_matrix.h5"
hd5 = sc.read_10x_h5(this_path)

# %% Testing with mtx
mtx_path = "processed_data/Spaceranger_uncompressed/V10F03-033_A/"
mtx = sc.read_10x_mtx(mtx_path, 'gene_ids')

# %% Testing with prebuilt visium
visium_data = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_A/outs/")
# %%
# From here https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html


visium_data.var["mt"] = visium_data.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(visium_data, qc_vars = ["mt"], inplace=True) # Where is the output?
sc.pl.spatial(visium_data, img_key="hires", color=["total_counts", "n_genes_by_counts"]) # Whoa!

# %% 
# Next steps
sc.pl.spatial(visium_data, img_key="hires", color="clusters", size=1.5)

# %%
