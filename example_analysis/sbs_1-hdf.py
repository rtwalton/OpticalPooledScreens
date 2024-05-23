import ops
from ops.imports import *

# Process cells files
ops.utils.combine_tables('cells',output_filetype='hdf',subdir_read='process_sbs/tables',n_jobs=-2, subdir_write='sbs_1')

# Process reads files
ops.utils.combine_tables('reads',output_filetype='hdf',subdir_read='process_sbs/tables',n_jobs=-2, subdir_write='sbs_1')

# Process sbs_info files
ops.utils.combine_tables('sbs_info',output_filetype='hdf',subdir_read='process_sbs/tables',n_jobs=-2, subdir_write='sbs_1')

# Process bases files -- not working currently
ops.utils.combine_tables('bases',output_filetype='hdf',subdir_read='process_sbs/tables',n_jobs=-2, subdir_write='sbs_1')