# %% Libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt # just for histograms!
# import plotnine as 
# %% Get Gene data 
# path_to_gene_outs_33c = "original_data/Spaceranger_analysis/V10F03-033_C/outs/"


# # Noncancerous
# gene_adata_33a = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_A/outs/")
# gene_adata_33a.var_names_make_unique(join = "-")

# gene_adata_33b = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_B/outs/")
# gene_adata_33b.var_names_make_unique(join= "-")



# # Cancerous
# gene_adata_33c = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_C/outs/")
# gene_adata_33c.var_names_make_unique(join = "-") # Make sure no duplicated gene names
# # Make obs names identifiable
# # gene_adata_33c.obs.index = ["33C_" + spot_name for spot_name in gene_adata_33c.obs.index] 

# gene_adata_33d = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_D/outs/")
# gene_adata_33d.var_names_make_unique(join = "-") # Make sure no duplicated gene names
# # Make obs names identifiable
# # gene_adata_33d.obs.index = ["33D_" + spot_name for spot_name in gene_adata_33d.obs.index]



# adatas = {
#     "33A" : gene_adata_33a,
#     "33B" : gene_adata_33b,
#     "33C" : gene_adata_33c,
#     "33D" : gene_adata_33d
# }

# %% Load in all data
main_config_paths = pd.read_csv("./image_inputters/main_config.csv") 
# gets 33A for V10F03-033_A
main_config_paths["readable_id"] = [x[3][:3] for x in 
                                    main_config_paths.loc[:,"true_annotation_path"].str.split("/")]

adatas = dict()
for row in main_config_paths.iterrows():
    single_adata = sc.read_visium(row[1]["spaceranger_path"] + "/outs/")
    single_adata.var_names_make_unique(join = "_")
    adatas[row[1]["readable_id"]] = single_adata


# %% Integrate annotations
paths_to_classification = "image_inputters/cleaned_classification"

for key in adatas.keys():
    classification_df = pd.read_csv(paths_to_classification + "/" + key + ".csv",
                                    names=  ["cluster", "classification", "int_class", "class_2"],
                                    index_col=0)
    classification_df.loc[:, "classification"].astype(str, copy=False)

    adatas[key].obs = adatas[key].obs.join(classification_df.loc[:,"classification"], how='left')

# %% Merge adatas

for key in adatas.keys():
    adatas[key].obs.index = [x + f'_{key}' for x in
        adatas[key].obs.index]

main_adata = ad.concat(adatas=adatas,
                      merge="same", 
                      label="dataset")
# new_adata.obs["dataset"] now tells us the dataset they each 
#   came from, or the end of the index
# %% Save main adata
main_adata.write("./intermediate_data/all_expression.h5ad")
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

# A and B are both meant to be control
normal_certains = [x and y for x,y in 
                 zip(main_adata.obs["dataset"].str.match("\d\d[AB]"),
                      main_adata.obs["classification"] == "normal")]
# C and D are both meant to be disease
disease_certains = [x and y for x,y in
                    zip(main_adata.obs["dataset"].str.match("\d\d[CD]"),
                        main_adata.obs["classification"] == "cancer")]
# Means of normal tissue
# -1 if x = 0 so that fold change works nicely
# will break log
nonc_means = main_adata[normal_certains].to_df().mean(axis=0).map(
    lambda x: -1 if x == 0 else x)

disease_adata = main_adata[disease_certains,:]

# Basic fc
# FC relative to mean of previous
disease_adata.layers["fold_change"] = disease_adata.to_df().div(nonc_means, axis='columns')

# disease_adata.layers["fold_change"].to_df()

# %% QC of data
qc_output = sc.pp.calculate_qc_metrics(main_adata)[0]
# Log1p is log(x+1) of values
plt.figure()
plt.hist(qc_output.loc[:,'n_genes_by_counts'])





# %% Qc of spots
# TODO: look at qc of spots


# TODO: run differential expression
# Being extra cautious
