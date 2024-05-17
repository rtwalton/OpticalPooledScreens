import ops
from ops.imports import *

# Process merge files
ops.utils.combine_tables('merge',output_filetype='hdf',subdir_read='alignment',n_jobs=-2, subdir_write='hash_3')