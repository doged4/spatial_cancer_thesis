# %% Libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt # just for histograms!
# import plotnine as 
# %% Get Gene data 
# path_to_gene_outs_33c = "original_data/Spaceranger_analysis/V10F03-033_C/outs/"


# Noncancerous
gene_adata_33a = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_A/outs/")
gene_adata_33a.var_names_make_unique(join = "-")

gene_adata_33b = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_B/outs/")
gene_adata_33b.var_names_make_unique(join= "-")



# Cancerous
gene_adata_33c = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_C/outs/")
gene_adata_33c.var_names_make_unique(join = "-") # Make sure no duplicated gene names
# Make obs names identifiable
# gene_adata_33c.obs.index = ["33C_" + spot_name for spot_name in gene_adata_33c.obs.index] 

gene_adata_33d = sc.read_visium("original_data/Spaceranger_analysis/V10F03-033_D/outs/")
gene_adata_33d.var_names_make_unique(join = "-") # Make sure no duplicated gene names
# Make obs names identifiable
# gene_adata_33d.obs.index = ["33D_" + spot_name for spot_name in gene_adata_33d.obs.index]

adatas = {
    "33A" : gene_adata_33a,
    "33B" : gene_adata_33b,
    "33C" : gene_adata_33c,
    "33D" : gene_adata_33d
}


new_adata = ad.concat(adatas=adatas,
                      merge="same", 
                      label="dataset")
# new_adata.obs["dataset"] now tells us the dataset they each came from
# note that obs names not unique now it seems

# ad.concat([gene_adata_33c, gene_adata_33d],)

# %% Look at gene expression levels
plt.figure()
plt.hist(new_adata[:,["BRCA1"]].to_df())
plt.title("BRCA1")

plt.figure()
plt.hist(new_adata[:,["BRCA2"]].to_df())
plt.title("BRCA2")

plt.figure()
plt.hist(new_adata[:,["TP53"]].to_df())
plt.title("TP53")

# %% Look at pathway participants
wkpathways = pd.read_table("external_data\wikipathways_breastcancer_participabts_WP4262.tsv")
wkp_genes = wkpathways.loc[wkpathways.loc[:,"Type"] == "GeneProduct","Label"] 
wkp_genes[18] = 'MRE11' # MRE11A --> MRE11 for convenience
wkp_genes = wkp_genes.unique()
# Might be better with ensembl ids in future

sum([x in new_adata.var.index for x in wkp_genes]) / len(wkp_genes)

plt.figure()
plt.hist(new_adata[:,wkp_genes].to_df().sum(axis = 1))
# new_adata[:,wkp_genes].to_df().nla
plt.title("Counts of Genes in BC Pathways")

plt.figure()
plt.hist(new_adata[:,wkp_genes].to_df().sum(axis = 1).nlargest(1000))
plt.title("Counts of Genes in BC Pathway top 1000")

print(new_adata[:,wkp_genes].to_df().sum(axis = 0).nlargest(100))

# %% Differential expression
