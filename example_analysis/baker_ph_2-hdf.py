from ops.utils import *

# Process sbs_info files
combine_tables('ph_info', output_filetype='hdf', subdir='process_ph/tables', n_jobs=48, output_dir = 'ph_1')

