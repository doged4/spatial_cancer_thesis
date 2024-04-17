# %% Libraries
import scanpy as sc
import anndata as ad
import pandas as pd

import logging 
from log_tools.StreamToLogger import StreamToLogger
import sys
import warnings


import matplotlib.pyplot as plt

import gseapy

# %% [markdown]
# This script is based on the `pathway_enrichment_tester.py` script.

# %% Get paths in st data

CONFIG_PATH = "classify/main_config.csv"
CLASSIFICATION_DIR = "intermediate_data/classification/cleaned_classification_wenwen"

DO_LOGGING = True

# %% Load in data [markdown]
# See `pathway_enricher.py` for original code

def retrieve_adata(slide_index, config_df, classifications_dir, suppress_warnings = False):
    """From a simple slide id, the main config df for metadata, and the classification dir, return adata."""
    simple_slide_id = config_df.loc[slide_index, 'simple_slide']

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            single_adata = sc.read_visium(config_df.loc[slide_index, "spaceranger_path"] + "/outs/")
    else:
        single_adata = sc.read_visium(config_df.loc[slide_index, "spaceranger_path"] + "/outs/")
        
    single_adata.uns['biopsy_sample_id'] = config_df.loc[slide_index, 'biopsy_sample_id'].replace('-','')
    single_adata.uns['biopsy_expected_classification'] = config_df.loc[slide_index, 'expected_classification']
    single_adata.uns['biopsy_annotated_classification'] = config_df.loc[slide_index, 'annotated_classification']
    single_adata.uns['patient'] = config_df.loc[slide_index, 'patient']

    # Read in spot classifications from pathologist annotations (see : image_inputters\cevi_altering_loupebrowser_parser.ipynb)
    spot_classifications = pd.read_csv(classifications_dir + "/" + simple_slide_id + ".csv",
                                    names=  ["cluster", "classification", "int_class", "class_2"],
                                    index_col=0)
    spot_classifications.loc[:, "classification"].astype(str, copy=False)
    slide_obs = single_adata.obs

    # Join spot info in image counts with classification
    slide_obs = slide_obs.join(spot_classifications.loc[:,"classification"], how='left')
    # Note that spots that cannot be found in classification table will be given: NA
    # Spots that were not certainly assigned by a pathologist recieved: " "
    
    # Set observations not 'cancer' or 'normal' to be 'unknown'
    slide_obs.loc[ 
        ~slide_obs.loc[:,"classification"].isin(['cancer', 'normal']), 
        "classification"] = 'unknown'
    
    
    # Column to track patient and slide classification for fully concatenated anndata operations
    slide_obs['patient'] = single_adata.uns['patient']
    slide_obs["slide_condition"] = single_adata.uns['biopsy_annotated_classification']
    slide_obs['biopsy_sample_id'] = single_adata.uns['biopsy_sample_id']
    
    slide_obs.index = [x + f'_{simple_slide_id}_{single_adata.uns["biopsy_sample_id"]}' for x in
        single_adata.obs.index]

    # Set back (may be unnecessary)
    single_adata.obs = slide_obs

    return single_adata

    
# %% Get gene sets
def get_de_genes(config_df, p_filter = None, n_filter = None, split_fc = False):
    """Return dict of significantly differentially expressed genes for sample in main config"""
    de_genes_dict = {}
    for biopsy, path in zip(config_df['simple_biopsy'], config_df['deg_path']):
        de_genes = pd.read_csv(path) # read from deg_path
        if 'gene' in de_genes.columns:
            if p_filter != None:
                de_genes = de_genes.loc[de_genes['p_val_adj'] <= p_filter,:]
            if split_fc:
                up_de_genes = de_genes.loc[de_genes['avg_log2FC'] > 0,:]
                dn_de_genes = de_genes.loc[de_genes['avg_log2FC'] < 0,:]
            if n_filter != None:
                if split_fc:
                    # Filters count once for each of them
                    up_de_genes.sort_values(by=['p_val_adj'], inplace=True)
                    up_de_genes = up_de_genes.iloc[:n_filter, :]
                    dn_de_genes.sort_values(by=['p_val_adj'], inplace=True)
                    dn_de_genes = dn_de_genes.iloc[:n_filter, :]
                    # print(sum(up_de_genes.duplicated()))
                    # print(sum(dn_de_genes.duplicated()))
                else:
                    de_genes.sort_values(by=['p_val_adj'], inplace=True)
                    de_genes = de_genes.iloc[:n_filter, :]
            if de_genes.shape[0] < 5:
                print(f"Be aware: {biopsy} seems to have only {de_genes.shape[0]}")
            if split_fc:
                de_genes_dict[f"{biopsy}_deg_set_up"] = list(up_de_genes.loc[:, 'gene'])
                de_genes_dict[f"{biopsy}_deg_set_dn"] = list(dn_de_genes.loc[:, 'gene'])
            else:
                de_genes_dict[f"{biopsy}_deg_set"] = list(de_genes.loc[:, 'gene'])
        else:
            print(f"Be aware: {biopsy} seems to have an empty gene list")
    return de_genes_dict


# %% Filter spots [markdown]
# Look at `spot_filter_testing.ipynb` for criteria
def filter_spots(gene_counts_ad, in_place_save = False):
    CRITERIA = 100
    """Filter spots according to greater than 100 counts per spot"""
    if in_place_save:
        qc_info, _ = sc.pp.calculate_qc_metrics(gene_counts_ad, inplace=False)
        passing_spots = qc_info.index[qc_info['total_counts'] > CRITERIA]
        return gene_counts_ad[passing_spots,:]
    else:
        sc.pp.calculate_qc_metrics(gene_counts_ad, inplace=True)
        return gene_counts_ad[gene_counts_ad.obs['total_counts'] > CRITERIA, :]
        
# %% Get enrichments
def get_enrichments_ad(gene_counts_adata, gene_set, verbose_log = True, sample_name = "", suppress_warnings = False):
    """Run ssGSEA on gene_counts_ad with enrichments for gene_set"""
    # TODO: get enrichments not to drop columns??
    if verbose_log:
        logging.debug(f"Starting enrichments for {sample_name}")
    # Get enrichments
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            all_ssgs_results = gseapy.ssgsea(
                    data=gene_counts_adata.to_df().T,
                    gene_sets=gene_set
                    )
    else:
        all_ssgs_results = gseapy.ssgsea(
                data=gene_counts_adata.to_df().T,
                gene_sets=gene_set
                )

    if verbose_log:
        logging.debug(f"Enrichments complete for {sample_name}")
        logging.debug("Converting to adata")
    # Collect results and save as anndata
    enrichments = all_ssgs_results.res2d.pivot(columns='Term',index='Name')
    
    enrichments = enrichments.astype('float')

    # Collect in adata
    enrichments_adata = ad.AnnData(
        X = enrichments['NES'], # save NORMALIZED
        obs=gene_counts_adata.obs,
        var = pd.DataFrame(data = {'term' : list(enrichments['ES'].columns)},
                            index = list(enrichments['ES'].columns)),
        uns=gene_counts_adata.uns,
        obsm=gene_counts_adata.obsm,
        layers={'nonnormalized' : enrichments['ES']}
        )
    if verbose_log:
        logging.debug("Enrichments as adata complete")
    # Return anndata
    return enrichments_adata



# %% 
TEST_MODE = False
# %% Block for testing
if TEST_MODE:
    config = pd.read_csv(CONFIG_PATH, index_col=0)
    config.set_index('simple_biopsy', drop = False, inplace = True)
    config['patient'] = [x[0] for x in config['biopsy_sample_id'].str.split(pat = "-")]

    gene_set = get_de_genes(config, p_filter=0.01)

    slide_id = list(config.index)[6]

    print(f"Retrieving adata {slide_id}")
    raw_genes_ad = retrieve_adata(slide_index=slide_id, config_df=config, classifications_dir=CLASSIFICATION_DIR)
    print("Adata retrieved")
    print("Filtering...")
    filtered_genes_ad = filter_spots(raw_genes_ad)
    print("Running enrichment")
    ssgsea_enrichments = get_enrichments_ad(filtered_genes_ad, gene_set = gene_set, verbose_log=DO_LOGGING, sample_name=slide_id)
    ssgsea_enrichments.write(f"intermediate_data/enrichments_on_de/{slide_id}_de_gene_enrichments.h5ad")

    print(f"Saved to intermediate_data/{slide_id}_bc_sig_enrichments_degenes.h5ad")

# %% 
BACKGROUND = False
SUPPRESS_WARNINGS = False
# %% Get gene counts
# %%
# OUT_DIR = "intermediate_data/enrichments_on_updn_de"
OUT_DIR = "intermediate_data/enrichments_on_external"
CANNONICAL_GENE_PATH = "intermediate_data/external_data/genesets.c6_breast_keyword.hsabiens.gmt"
if __name__ == '__main__':
    
    if DO_LOGGING:
        log = logging.getLogger('log_all')
        sys.stdout = StreamToLogger(log, logging.INFO, background=BACKGROUND)
        sys.stderr = StreamToLogger(log, logging.ERROR, background=BACKGROUND)

        logging.basicConfig(filename= "pathway_enrichments.log", 
                            format='%(asctime)s - %(message)s', 
                            level=logging.DEBUG, filemode='a')
        logging.debug("Starting Main run")
    
        
    config = pd.read_csv(CONFIG_PATH, index_col=0)
    config.set_index('simple_biopsy', drop = False, inplace = True)
    config['patient'] = [x[0] for x in config['biopsy_sample_id'].str.split(pat = "-")]

    # FILTER PATIENT LIST
    config = config.loc[config.loc[:, 'annotated_classification'] == 'Tumor' ,:]
    print("Patient list here")
    print(config)

    # gene_set = get_de_genes(config, p_filter=0.01)
    # gene_set = get_de_genes(config, n_filter=100, split_fc=True)
    print("Using this gene set")
    if CANNONICAL_GENE_PATH != "":
        gene_set = CANNONICAL_GENE_PATH
        print(gene_set)
    else:
        print(gene_set.keys())

    if DO_LOGGING:
        logging.debug("Starting enrichments")
    
    for slide_id in list(config.index):
        if DO_LOGGING:
            logging.debug(f"Retrieving adata {slide_id}")
        raw_genes_ad = retrieve_adata(slide_index=slide_id, config_df=config, classifications_dir=CLASSIFICATION_DIR, suppress_warnings=SUPPRESS_WARNINGS)
        if DO_LOGGING:
            logging.debug("Adata retrieved")
            logging.debug("Filtering...")
        filtered_genes_ad = filter_spots(raw_genes_ad)
        if DO_LOGGING:
            logging.debug("Running enrichment")
        ssgsea_enrichments = get_enrichments_ad(filtered_genes_ad, gene_set = gene_set, verbose_log=DO_LOGGING, sample_name=slide_id, suppress_warnings=SUPPRESS_WARNINGS)
        # ssgsea_enrichments.write(f"{OUT_DIR}/{slide_id}_de_gene_enrichments.h5ad")
        ssgsea_enrichments.write(f"{OUT_DIR}/{slide_id}_gene_enrichments.h5ad")
        if DO_LOGGING:
            # logging.debug(f"Saved to intermediate_data/{slide_id}_bc_sig_enrichments_degenes.h5ad")
            logging.debug(f"Saved to {OUT_DIR}/{slide_id}_gene_enrichments.h5ad")
    if DO_LOGGING:
        logging.debug(f"Run completed")