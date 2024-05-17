import ops
from ops.utils import *

# Process phenotype_info files
ops.utils.combine_tables('phenotype_info',output_filetype='hdf',subdir_read='process_ph/tables',n_jobs=-2, subdir_write='ph_2')

# Process cp_phenotype files with minimal information
ops.utils.combine_tables('cp_phenotype',output_filetype='hdf',subdir_read='process_ph/tables',n_jobs=-2,
                         usecols=['well','tile','label',
                                  'cell_i','cell_j','cell_bounds_0','cell_bounds_1','cell_bounds_2','cell_bounds_3',
                                  'cell_dapi_min','cell_cenpa_min','cell_coxiv_min','cell_wga_min'],
                         subdir_write='ph_2')

# Rename minimal cp_phenotype file
os.rename('ph_2/cp_phenotype.hdf','ph_2/min_cp_phenotype.hdf')

# Process cp_phenotype files 
ops.utils.combine_tables('cp_phenotype',output_filetype='hdf',subdir_read='process_ph/tables',n_jobs=-2, subdir_write='ph_2')