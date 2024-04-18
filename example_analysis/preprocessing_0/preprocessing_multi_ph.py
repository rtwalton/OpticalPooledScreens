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

# Set input directories for PH
input_directory_ph = os.path.join(home, 'input_ph/process/input/20X/')

# Specify the channel orders for PH
channel_order_ph = ['DAPI 1x1 LF', 'GFP 1x1 LF', 'A594 1x1 LF', 'A750_1x1_LF']

# Convert PH images to multidimensional TIFFs
convert_to_multidimensional_tiff_ph(input_directory_ph, channel_order_ph)
