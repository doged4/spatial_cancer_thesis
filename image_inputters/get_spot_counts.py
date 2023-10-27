# Author: Cevi Bainton
# Notes: Some below code is from a spaceranger help page
#        10x genomics code is sourced from https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/advanced/h5_matrices#hdf5-file
#%% Libraries
import collections
import scipy.sparse as sp_sparse
import scipy.io as sio
import tables

import os
from pandas import DataFrame
 
#%% Space Ranger code
# The below code was taken from 10X genomics SpaceRanger help page

CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
 
def get_matrix_from_h5(filename):
    """
    Return matrix from path to hdf5 filename for a SpaceRanger counts file
    """
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
         
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        bin_tag_keys = getattr(feature_group, '_all_tag_keys').read() # convert tag keys from b'word' --> 'word'

        tag_keys = [str(x, encoding="utf-8") for x in bin_tag_keys]

        for key in tag_keys:
            feature_ref[key] = getattr(feature_group, key).read()
        
        return CountMatrix(feature_ref, barcodes, matrix)
 
filtered_h5 = r"\Users\cbainton\Desktop\ST_project\original_data\Spaceranger_analysis\V10F03-033_A\outs\filtered_feature_bc_matrix.h5"
filtered_matrix_h5 = get_matrix_from_h5(filtered_h5)
# %%
for filtered_h5_path in sample_list:
    this_countmatrix = get_matrix_from_h5(filtered_h5_path)
    feature_ref = DataFrame(this_countmatrix.feature_ref)
    barcodes = DataFrame(this_countmatrix.barcodes)
    
    feature_ref.to_csv(os.path.join(prefix, "features.tsv.gz"), sep="\t", compression="gzip")
    barcodes.to_csv(os.path.join(prefix, "barcodes.tsv.gz"), sep="\t", compression="gzip")

    sio.mmwrite(os.path.join(prefix, "matrix.mtx"), this_countmatrix.matrix) # write sparse matrix
