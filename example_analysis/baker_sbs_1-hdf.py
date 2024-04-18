from ops.utils import *

# Process cells files
combine_tables('cells', output_filetype='hdf', subdir='process_sbs/tables', n_jobs=48, output_dir = 'sbs_1')

# Process reads files
combine_tables('reads', output_filetype='hdf', subdir='process_sbs/tables', n_jobs=48, output_dir = 'sbs_1')

# Process sbs_info files
combine_tables('sbs_info', output_filetype='hdf', subdir='process_sbs/tables', n_jobs=48, output_dir = 'sbs_1')

# Process bases files
combine_tables('bases', output_filetype='csv', subdir='process_sbs/tables', n_jobs=48, output_dir = 'sbs_1')