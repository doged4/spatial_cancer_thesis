
# %% Test with h5
import scanpy as sc

this_path = "original_data/Spaceranger_analysis/V10F03-033_A/outs/filtered_feature_bc_matrix.h5"
hd5 = sc.read_10x_h5(this_path)

# %% Testing with mtx
mtx_path = "processed_data/Spaceranger_uncompressed/V10F03-033_A/"
mtx = sc.read_10x_mtx(mtx_path, 'gene_ids')

# %% Testing with prebuilt visium
visium_data = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_C/outs/")
# %%
# From here https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html


visium_data.var["mt"] = visium_data.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(visium_data, qc_vars = ["mt"], inplace=True) # Where is the output?
sc.pl.spatial(visium_data, img_key="hires", color=["total_counts", "n_genes_by_counts"]) # Whoa!

# %%
sc.pp.normalize_total(visium_data, inplace=True)
sc.pp.log1p(visium_data)
sc.pp.highly_variable_genes(visium_data, flavor="seurat", n_top_genes=2000)


sc.pp.pca(visium_data)
sc.pp.neighbors(visium_data)
sc.tl.umap(visium_data)
sc.tl.leiden(visium_data, key_added="clusters")  #clusters in the umap?

# %% 
# Next steps
sc.pl.spatial(visium_data, img_key="hires", color="clusters", size=1.5)

# %%

type(visium_data.raw)


# %% Image work

visium_data = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_C/outs/") #,
                            #  source_image_path="original_data/original_data/High-resolution_tissue_images/V10F03-033/201210_BC_V10F03-033_S8C-T_RJ.C1-Spot000001.jpg")

# Below found with 
# sc.pp.highly_variable_genes(visium_data, flavor="seurat", n_top_genes=10)
# visium_data.var['highly_variable'][visium_data.var['highly_variable']]

sc.pl.spatial(visium_data, img_key="hires", color = "LINC00632")
#%% Get df from adata
visium_data.to_df()

# %%
# Look here next! : https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html
# img.generate_spot_crops looks crazy!!


# We want: how to best get data from AnnData --> tensor
# Later steps:
#       image matrices --> tensor
# or get things in anndata