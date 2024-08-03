import os
import re
from natsort import natsorted
from collections import OrderedDict, Counter, defaultdict
from functools import partial
import glob
from itertools import product
import warnings
import numpy as np
import pandas as pd
import skimage

# Suppress specific warnings related to numpy dtype size changes
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Local module imports
from .annotate import (
    annotate_labels,       # Function to annotate labels
    annotate_points,       # Function to annotate points
    annotate_bases,        # Function to annotate bases
    GRMC                   # GRMC constant or class
)

from .io import (
    BLUE,                  # Color constant
    GREEN,                 # Color constant
    RED,                   # Color constant
    MAGENTA,               # Color constant
    GRAY,                  # Color constant
    CYAN,                  # Color constant
    GLASBEY,               # Color map
    grid_view               # Function for grid view
)

from .io import (
    read_stack as read,    # Alias for function to read stack
    save_stack as save     # Alias for function to save stack
)

from .filenames import (
    name_file as name,     # Alias for function to generate filenames
    parse_filename as parse,  # Alias for function to parse filenames
    timestamp,             # Function or variable for timestamp
    file_frame             # Function or variable for file frame
)

from .utils import (
    or_join,               # Function to join with OR logic
    and_join,              # Function to join with AND logic
    groupby_reduce_concat, # Function for grouping, reducing, and concatenating
    groupby_histogram,     # Function for creating histograms from groups
    replace_cols,          # Function to replace columns in a DataFrame
    pile,                  # Function for stacking or piling data
    montage,               # Function for creating montages
    make_tiles,            # Function for making tiles
    trim,                  # Function to trim data or images
    join_stacks,           # Function to join stacks
    csv_frame              # Function for creating a DataFrame from CSV
)

from . import in_situ      # Import the in_situ module
