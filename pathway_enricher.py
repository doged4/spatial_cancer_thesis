# %% Libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt # just for histograms!
import numpy as np
import math

# from IPython.display import display, HTML
import plotnine as pn


# %% Load in all data
# Get paths
main_config = pd.read_csv("./image_inputters/main_config.csv") 
# Gets slide id like 33A for V10F03-033_A
main_config["readable_id"] = [x[-1][:3] for x in 
                                    main_config.loc[:,"true_annotation_path"].str.split("/")]

# Get patient id
if sum(main_config['biopsy_sample_id'].isna()) > 0:
    main_config.loc[main_config['biopsy_sample_id'].isna(),'biopsy_sample_id'] = 'unknown-x-x'
    raise RuntimeWarning("There appears to be a nonlabelled sample")

main_config['patient'] = [x[0] for x in main_config['biopsy_sample_id'].str.split(pat = "-")]
main_config.set_index('readable_id', drop=False, inplace=True)

# Load in adatas
adatas = dict()
for row in main_config.iterrows():
    single_adata = sc.read_visium(row[1]["spaceranger_path"] + "/outs/")
    single_adata.var_names_make_unique(join = "_")

    # Keep track of attributes of each sample in the unstructured region
    single_adata.uns['slide_id'] = row[1]['slide_id']
    single_adata.uns['biopsy_sample_id'] = row[1]['biopsy_sample_id'].replace('-','')
    single_adata.uns['biopsy_expected_classification'] = row[1]['expected_classification']
    single_adata.uns['biopsy_annotated_classification'] = row[1]['annotated_classification']
    single_adata.uns['patient'] = row[1]['patient']
    
    # Save to dict
    adatas[row[1]["readable_id"]] = single_adata


# Integrate annotations from pathologist
paths_to_classification = "intermediate_data/classification/cleaned_classification_wenwen"

for key in adatas.keys():
    # Read in spot classifications from pathologist annotations (see : image_inputters\cevi_altering_loupebrowser_parser.ipynb)
    spot_classifications = pd.read_csv(paths_to_classification + "/" + key + ".csv",
                                    names=  ["cluster", "classification", "int_class", "class_2"],
                                    index_col=0)
    spot_classifications.loc[:, "classification"].astype(str, copy=False)

    slide_obs = adatas[key].obs

    # Join spot info in image counts with classification
    slide_obs = slide_obs.join(spot_classifications.loc[:,"classification"], how='left')
    # Note that spots that cannot be found in classification table will be given: NA
    # Spots that were not certainly assigned by a pathologist recieved: " "

    # Add category `unkown`
    # slide_obs['classification'] = slide_obs['classification'].cat.add_categories('unknown')

    # Set observations not 'cancer' or 'normal' to be 'unknown'
    slide_obs.loc[ 
        ~slide_obs.loc[:,"classification"].isin(['cancer', 'normal']), 
        "classification"] = 'unknown'
    
    
    # Column to track patient and slide classification for fully concatenated anndata operations
    slide_obs['patient'] = adatas[key].uns['patient']
    slide_obs["slide_condition"] = adatas[key].uns['biopsy_annotated_classification']
    slide_obs['biopsy_sample_id'] = adatas[key].uns['biopsy_sample_id']
    # Set back (may be unnecessary)
    adatas[key].obs = slide_obs

# Merge adatas
for key in adatas.keys():
    adatas[key].obs.index = [x + f'_{key}_{adatas[key].uns["biopsy_sample_id"]}' for x in
        adatas[key].obs.index]

main_adata = ad.concat(adatas=adatas,
                      merge="same", 
                      label="dataset")
# note uns data is lost here

# new_adata.obs["dataset"] now tells us the slide id that they each 
#   came from, or the end of the index
# %% Save main adata
# main_adata.write("./intermediate_data/all_expression.h5ad")
# main_adata = ad.read_h5ad("./intermediate_data/all_expression.h5ad")

# %% Look at gene expression levels
plt.figure()
plt.hist(main_adata[:,["BRCA1"]].to_df())
plt.title("BRCA1")

plt.figure()
plt.hist(main_adata[:,["BRCA2"]].to_df())
plt.title("BRCA2")

plt.figure()
plt.hist(main_adata[:,["TP53"]].to_df())
plt.title("TP53")

# %% Look at pathway participants
wkpathways = pd.read_table("intermediate_data\external_data\wikipathways_breastcancer_participabts_WP4262.tsv")
wkp_genes = wkpathways.loc[wkpathways.loc[:,"Type"] == "GeneProduct","Label"] 
wkp_genes[18] = 'MRE11' # MRE11A --> MRE11 for convenience
wkp_genes = wkp_genes.unique()
# Might be better with ensembl ids in future

sum([x in main_adata.var.index for x in wkp_genes]) / len(wkp_genes)

plt.figure()
plt.hist(main_adata[:,wkp_genes].to_df().sum(axis = 1))
# new_adata[:,wkp_genes].to_df().nla
plt.title("Counts of Genes in BC Pathways")

plt.figure()
plt.hist(main_adata[:,wkp_genes].to_df().sum(axis = 1).nlargest(1000))
plt.title("Counts of Genes in BC Pathway top 1000")

print(main_adata[:,wkp_genes].to_df().sum(axis = 0).nlargest(100))

# %% Differential expression starts

# # A and B are both meant to be control
# normal_certains = [x and y for x,y in 
#                  zip(main_adata.obs["dataset"].str.match("\d\d[AB]"),
#                       main_adata.obs["classification"] == "normal")]
# # C and D are both meant to be disease
# disease_certains = [x and y for x,y in
#                     zip(main_adata.obs["dataset"].str.match("\d\d[CD]"),
#                         main_adata.obs["classification"] == "cancer")]
# # Means of normal tissue
# # -1 if x = 0 so that fold change works nicely
# # will break log
# nonc_means = main_adata[normal_certains].to_df().mean(axis=0).map(
#     lambda x: -1 if x == 0 else x)

# disease_adata = main_adata[disease_certains,:]

# # Basic fc
# # FC relative to mean of previous
# disease_adata.layers["fold_change"] = disease_adata.to_df().div(nonc_means, axis='columns')

# # disease_adata.layers["fold_change"].to_df()

# %% QC of data
qc_output = sc.pp.calculate_qc_metrics(main_adata)[0]
# Log1p is log(x+1) of values
plt.figure()
plt.hist(qc_output.loc[:,'log1p_n_genes_by_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_n_genes_by_counts'], bins = 100)
# plt.legend()
plt.title('Log1p genes per count: blue is all, orange is just disease')

plt.figure()
plt.hist(qc_output.loc[:,'log1p_total_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_total_counts'], bins = 100)
# plt.legend()
plt.title('Log1p counts per cell: blue is all, orange is just disease')
qc_output['total_counts'].quantile([0,0.25,0.5,0.75,1])

# %% What to filter by?
print(qc_output.filter(regex="C\d$",axis=0)['total_counts'].quantile([0,0.25,0.5,0.75,1]))
print(qc_output.filter(regex="T\d$",axis=0)['total_counts'].quantile([0,0.25,0.5,0.75,1]))

print(qc_output['total_counts'].quantile([0.1]))
# %%[markdown] This seems good for counts
# Bottom 10% looks like those with counts of 46 or so, which is very little.
# We'll filter for this
# %% 
print(qc_output.filter(regex="C\d$",axis=0)['n_genes_by_counts'].quantile([0,0.25,0.5,0.75,1]))
print(qc_output.filter(regex="T\d$",axis=0)['n_genes_by_counts'].quantile([0,0.25,0.5,0.75,1]))

print(qc_output['n_genes_by_counts'].quantile([0.1]))


#%%[markdown]
# ## Filtering by spot data
# We plan to filter the data as follows:
# 0. Batch effect correction?
# 1. Filter poorly annotated cells
#       * Filter cells that have annotations not matching their slide environments
# 2. Filter out cells with low gene counts
# 3. Filter out genes that appear in very few cells
# 4. Filter for genes that are highly variable

# %% Qc of spots
# TODO: look at qc of spots
filtered_adata = main_adata.copy()
pre_qc_output = sc.pp.calculate_qc_metrics(filtered_adata)[0]
print(f"Total count quartiles\n{pre_qc_output['total_counts'].quantile([0,0.25,0.5,0.75,1])}")
print(f"N genes by counts quartiles\n{pre_qc_output['n_genes_by_counts'].quantile([0,0.25,0.5,0.75,1])}")


# ##[markdown]
# We need to filter down to data to spots we are confident in. 
# The author notes that the terms "cancer" and "dcis" are 

# %% Filter for cells within our annotations
# Filter for having annotation
filtered_adata = filtered_adata[
    filtered_adata.obs["classification"].isin([
        "normal",
        "cancer",
        "dcis"
    ]), :]
# This removes " " and Na values for unsure and non annotated spots

# Confirm annotations match expected slide condition
# A and B are meant to be control
control_certains = [x == "Normal" and y == "normal" for x,y in 
                 zip(filtered_adata.obs["slide_condition"],
                      filtered_adata.obs["classification"])]
# C and D are both meant to be disease
disease_certains = [x == "Tumor" and y in ["cancer", "dcis"] for x,y in
                    zip(filtered_adata.obs["slide_condition"],
                        filtered_adata.obs["classification"])]
id_certains = [x or y for x,y in zip(control_certains, disease_certains)]

filtered_adata = filtered_adata[id_certains, :]


# %% Check qc info
qc_output = sc.pp.calculate_qc_metrics(filtered_adata)[0]
# Log1p is log(x+1) of values
plt.figure()
plt.hist(qc_output.loc[:,'log1p_n_genes_by_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_n_genes_by_counts'], bins = 100)
# plt.legend()
plt.title('Log1p genes per count: blue is all, orange is just disease')

plt.figure()
plt.hist(qc_output.loc[:,'log1p_total_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_total_counts'], bins = 100)
# plt.legend()
plt.title('Log1p counts per cell: blue is all, orange is just disease')
qc_output['total_counts'].quantile([0,0.25,0.5,0.75,1])


# %%[markdown]
# ## Merging Replicates
# We can combine the slides from the same patient with the same disease state. This means we are _ignoring_ potential batch effects. These will hopefully be captured by 

# %% Merging  replicates
# Merge replicate slides
# TODO: confirm below works

control_adatas = dict()
disease_adatas = dict()
# Combine each adata with those of same patient and slide level classification
for row in main_config.iterrows():
    patient_id = row[1]["patient"]
    slide_id = row[1]["readable_id"]
    # Exclude the slide with unkown patient id
    if patient_id =='unknown':
        continue
    # Decide target ID based on total slide classification
    if row[1]['annotated_classification'] == 'Normal':
        target_dict = control_adatas
    elif row[1]['annotated_classification'] == 'Tumor':
        target_dict = disease_adatas
    else:
        continue
    # Merge adata based on tec
    if patient_id in target_dict.keys():
        target_dict[patient_id] = ad.concat(adatas=[target_dict[patient_id], adatas[slide_id]],
                                merge="same", 
                                label="dataset")
    else:
        target_dict[patient_id] = adatas[slide_id]


#%%[markdown]
# This brings together our technical replicates for each patient. We can see that they have differing total spots that pass filter
#%%
print("Disease spots")
print([(sample, disease_adatas[sample].shape[0]) for sample in disease_adatas])
print("Control spots")
print([(sample, control_adatas[sample].shape[0]) for sample in control_adatas])
# %%[markdown]
# ## Filtering
# We will be careful as we transition into filtering spots by the quality of their annotations and amount of captured genes.
# ### Filtering by Annotation
# Some spots were not annotated as being clearly cancer or clearly control even if they were in a slide that expected this.
# We will remove spots that were not classified. We will also remove those that were classified as noncancerous in the slides that expected to find tumor. These spots are normal tissue adjacent to the tumor (NAT). To simplify our analysis so that the microenvironment of our controls is more constant, we shall remove these tissues.
#%% See annotation fracs
annotation_colors = {'cancer':"#F8776D",
                     'dcis':'orange',
                     'normal':'#5090FF',
                     'unclassified':'#555555'
                     }
disease_class_stats = pd.concat(
    [disease_adatas[sample].obs.loc[:,['patient','classification']] 
     for sample in disease_adatas.keys()], axis=0
)
disease_class_stats['classification'] =  disease_class_stats['classification'].apply(
        lambda x: x if x in ["normal", "cancer", "dcis"] else "unclassified")

(pn.ggplot(disease_class_stats, pn.aes(x="factor(patient)", fill="factor(classification)")) +
  pn.geom_bar() +
   pn.xlab("Patient") +
   pn.ylab("Spot count") + 
   pn.ggtitle("Classification in Disease Slides") +
   pn.scale_fill_manual(annotation_colors) +
   pn.guides(fill = pn.guide_legend(title="Classification")) )
#%% [markdown]
# We can see that patients 35, 70, and 71 will have to excluded from the analysis
#%% See in control
control_class_stats = pd.concat(
    [control_adatas[sample].obs.loc[:,['patient','classification']] 
     for sample in control_adatas.keys()], axis=0
)
control_class_stats['classification'] =  control_class_stats['classification'].apply(
        lambda x: x if x in ["normal", "cancer", "dcis"] else "unclassified")

(pn.ggplot(control_class_stats, pn.aes(x="factor(patient)", fill="factor(classification)")) +
  pn.geom_bar() +
   pn.xlab("Patient") +
   pn.ylab("Spot count") + 
   pn.ggtitle("Classification in Control Slides") +
   pn.scale_fill_manual(annotation_colors) +
   pn.guides(fill = pn.guide_legend(title="Classification")))
# %% Filter by annotation
# Filter to most certain annotations control
for key in control_adatas.keys():
    control_adatas[key] = control_adatas[key][
        control_adatas[key].obs['classification'] == "normal", :
    ]
# Filter to most certain annotations disease
for key in disease_adatas.keys():
    disease_adatas[key] = disease_adatas[key][
        disease_adatas[key].obs['classification'].isin(["cancer", "dcis"]), :
    ]
#%%[markdown]
# %% Filter for cell quality and normalize
def basic_filter_adata(adata):
    """Perform preliminary steps from Seurats filtering process. 
    See _recipes of Scanpy for details.
    Args: adata, and AnnData datatype
            expects adata.X as non logged data
    Returns: None, occurs in place"""
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # This does not include Seurat's filter by gene dispersion, rescaling

# Filter to most certain annotations control
for key in control_adatas.keys():
    basic_filter_adata(control_adatas[key])

# Filter to most certain annotations disease
for key in disease_adatas.keys():
    basic_filter_adata(disease_adatas[key])


#%%[markdown]
# Our data are now in control-disease pair dictionaries. We can use log1p and do differential expression

#%% REMOVE UNUSABLE COMPARISONS
# TODO: formalize removal of samples
# Temporary code for the deletions as seen above
del disease_adatas['35']
del disease_adatas['70']
del disease_adatas['71']

#%%Differential expression between pairs
for key in disease_adatas.keys():
    print(disease_adatas[key].shape)

    # Get genes that passed Seurat params in both controls and disease
    disease_genes = disease_adatas[key].var.index
    control_genes = control_adatas[key].var.index
    shared_genes = disease_genes[
        [x in control_genes for x in disease_genes]]
    print(f"Set {key} shares {len(shared_genes)} genes")
    # Select only shared genes
    disease_adatas[key] = disease_adatas[key][:, shared_genes]
    control_adatas[key] = control_adatas[key][:, shared_genes]

    # Make array of log1p means for normalization
    control_means = control_adatas[key].to_df().mean(axis=0)
    control_means = control_means.map(math.log1p)
    control_means = np.array(control_means)

    # Convert normalized data to log1p space
    sc.pp.log1p(disease_adatas[key])
    print(f"Normalizing {key}")
    # Subtract log1p of means from log1p of our data, as a matrix
    rows_count = disease_adatas[key].shape[0]
    disease_adatas[key].X = disease_adatas[key].X - np.broadcast_to(control_means, (rows_count, len(control_means)))

# %% [markdown]
# Next step is to find statistical significances of this differential expression

# TODO: justify the below parameters / convert to Seurat or Cell ranger values
# %% Run our filtering
print(qc_output['n_genes_by_counts'].quantile([0,0.25,0.5,0.75,1]))
# We will use the Seurat parameters
sc.pp.filter_cells(filtered_adata, min_genes=200)
sc.pp.filter_genes(filtered_adata, min_cells=3)

new_qc_output = sc.pp.calculate_qc_metrics(filtered_adata)[0]

plt.figure()
plt.hist(new_qc_output.loc[:,'log1p_n_genes_by_counts'], bins = 100)
plt.hist(new_qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_n_genes_by_counts'], bins = 100)
# plt.legend()
plt.title('Log1p genes per count: blue is all, orange is just disease')

plt.figure()
plt.hist(new_qc_output.loc[:,'log1p_total_counts'], bins = 100)
plt.hist(new_qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_total_counts'], bins = 100)
# plt.legend()
plt.title('Log1p counts per cell: blue is all, orange is just disease')

# %% Save for testing elsewhere
adatas['33D'].write("./intermediate_data/33D_S8T2.h5ad")
adatas['33B'].write("./intermediate_data/33B_S8C2.h5ad")

# %% Run Satija (Seurat) filterer
# Filter genes for at least a count
# Satija runs the following filters behind the scenes
#   pp.filter_cells(adata, min_genes=200)
#   pp.filter_genes(adata, min_cells=3)
#   normalize_total(adata, target_sum=1e4)
#   filter_result = filter_genes_dispersion(
#           adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5, log=not log
#       )
#   # filter genes in place
#   pp.scale(adata, max_value=10) # Will give mean 0, std 1, anything 10 std from mean is cut



# # Filter for minimum number of counts
# sc.pp.filter_cells(filtered_adata, 
#                    min_counts=46)
# # Filter for minimum number of read genes
# sc.pp.filter_cells(filtered_adata,
#                    min_genes=42)

# %% Run zheng filterer

# %%[markdown]
# We will now isolate the most variable genes.

# %% Isolate most variable genes


# %%
sc.pp.neighbors(filtered_adata)
sc.tl.umap(filtered_adata)
sc.pl.umap(filtered_adata, color = 'dataset')

sc.pl.umap(filtered_adata, color = 'patient')


# %% Post QC of data
qc_output = sc.pp.calculate_qc_metrics(filtered_adata)[0]
# Log1p is log(x+1) of values
plt.figure()
plt.hist(qc_output.loc[:,'log1p_n_genes_by_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_n_genes_by_counts'], bins = 100)
# plt.legend()
plt.title('Log1p genes per count: blue is all, orange is just disease')

plt.figure()
plt.hist(qc_output.loc[:,'log1p_total_counts'], bins = 100)
plt.hist(qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_total_counts'], bins = 100)
# plt.legend()
plt.title('Log1p counts per cell: blue is all, orange is just disease')
qc_output['total_counts'].quantile([0,0.25,0.5,0.75,1])

# %% What genes are we actually seeing?
sc.pl.highest_expr_genes(filtered_adata)
# ##[markdown]
# It appears as though we see many mitochondrial gene with some with very high expression

# %% PCA of main data
sc.pp.pca(filtered_adata, n_comps=50)
sc.pl.pca(filtered_adata, 
          annotate_var_explained=True, 
          color = 'dataset')

sc.pl.pca(filtered_adata, 
          annotate_var_explained=True, 
          color = 'classification')


# %% Trying Zhang filtering
zfiltered_adata = main_adata.copy()
sc.pp.filter_cells(zfiltered_adata,min_counts=1)
sc.pp.recipe_zheng17(zfiltered_adata, n_top_genes=2000, plot=True)
# qc_output = sc.pp.calculate_qc_metrics(zfiltered_adata)[0]
# ##[markdown]
# This seems to filter down to 2000 genes, and it focusses on the dispersions of genes to filter by

sc.pp.pca(zfiltered_adata, n_comps=50)
sc.pl.pca(zfiltered_adata, 
          annotate_var_explained=True, 
          color = 'dataset')

sc.pl.pca(zfiltered_adata, 
          annotate_var_explained=True, 
          color = 'classification')
# %% Seurat style filter
sfiltered_adata = main_adata.copy()
sc.pp.recipe_seurat(sfiltered_adata,
                    log=True, # means we are putting in NOT logged data
                    plot=True)
# Runs the following filters behind the scenes
#   pp.filter_cells(adata, min_genes=200)
#   pp.filter_genes(adata, min_cells=3)
#   normalize_total(adata, target_sum=1e4)
#   pp.scale(adata, max_value=10) # Will give mean 0, std 1, anything 10 std from mean is cut





seurat_qc_output = sc.pp.calculate_qc_metrics(sfiltered_adata)[0]
# Log1p is log(x+1) of values
plt.figure()
plt.hist(seurat_qc_output.loc[:,'log1p_n_genes_by_counts'], bins = 100)
plt.hist(seurat_qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_n_genes_by_counts'], bins = 100)
# plt.legend()
plt.title('Log1p genes per count: blue is all, orange is just disease')

plt.figure()
plt.hist(seurat_qc_output.loc[:,'log1p_total_counts'], bins = 100)
plt.hist(seurat_qc_output.filter(regex="T\d$", axis = 0).loc[:,'log1p_total_counts'], bins = 100)
# plt.legend()
plt.title('Log1p counts per cell: blue is all, orange is just disease')
seurat_qc_output['total_counts'].quantile([0,0.25,0.5,0.75,1])

# %% Checking seurat
# sc.pp.pca(sfiltered_adata, n_comps=50)
# sc.pl.pca(sfiltered_adata, annotate_var_explained=True, color = 'dataset')

sc.pp.neighbors(sfiltered_adata)
sc.tl.umap(sfiltered_adata)
sc.pl.umap(sfiltered_adata, color = 'dataset')

sc.pl.umap(sfiltered_adata, color = 'patient')
sc.pl.umap(sfiltered_adata, color='classification')


# TODO: filter based on whole slide quality
# --TODO: filter based on cell counts and gene counts--
# TODO: run differential expression
# Then GSEA!
# Being extra cautious


# %% Running gprofiler with scanpy

# Never run yet
# sc.queries.enrich(adata,
#                   group = ,
#                   key = , 
#                   org = 'hsapiens')


# %%
