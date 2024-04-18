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

# Set input directories for SBS
input_directory_sbs = os.path.join(home, 'input_sbs/process/input/10X/')

# Specify the channel orders for SBS
channel_order_sbs = ['Dapi_1p', 'CY3_30p_545', 'A594_30p', 'CY5_30p', 'CY7_30p']

# Convert SBS images to multidimensional TIFFs
convert_to_multidimensional_tiff_sbs(input_directory_sbs, channel_order_sbs)
