import os
import glob
from ops.imports import *
from ops.nd2_to_tif import *

# Set screen directories
parse_function_home = "/lab/barcheese01/screens"
parse_function_dataset = "baker"

# Set home directory as a combination of parse_function_home and parse_function_dataset
home = os.path.join(parse_function_home, parse_function_dataset)

# Change to the home directory
os.chdir(home)

# Get relevant ND2 files
file_list = glob.glob(os.path.join(home, '**/*.nd2'), recursive=True)
filtered_file_list = [file_path for file_path in file_list if "6W_centers" not in file_path]

# Extract metadata
extract_and_save_metadata(filtered_file_list, 
                          parse_function_home=parse_function_home, 
                          parse_function_dataset=parse_function_dataset)

# Extract single-channel TIFF files
parallel_convert(filtered_file_list, n_jobs=12, 
                 parse_function_home=parse_function_home, 
                 parse_function_dataset=parse_function_dataset)

