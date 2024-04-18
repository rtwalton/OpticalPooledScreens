import os
import re
from natsort import natsorted
from collections import OrderedDict, Counter, defaultdict
from functools import partial
import glob
from itertools import product

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd
import skimage

from .annotate import annotate_labels, annotate_points, annotate_bases, GRMC
from .io import BLUE, GREEN, RED, MAGENTA, GRAY, CYAN, GLASBEY, grid_view
from .io import read_stack as read, save_stack as save
from .filenames import (
    name_file as name,
    parse_filename as parse,
    timestamp,
    file_frame,
)
from .nd2_to_tif import (
    extract_and_save_metadata,
    convert_to_multidimensional_tiff_ph,
    convert_to_multidimensional_tiff_sbs,
    convert_to_tif,
    parallel_convert,
    extract_cycle,
    extract_well,
    extract_tile,
    extract_plate,
    extract_channel,
    parse_file,
)

# from .nd2_common import (
#     add_neighbors,
#     parse_nd2_filename,
#     extract_nd2_metadata_py,
#     get_metadata_at_coords,
#     get_axis_size,
#     extract_nd2_metadata_sdk,
#     build_file_table,
#     export_nd2_sdk_file_table,
#     read_nd2,
#     export_nd2,
# )
# from .preprocess_nd2s import process_files

from .utils import (
    or_join,
    and_join,
    groupby_reduce_concat,
    groupby_histogram,
    replace_cols,
    pile,
    montage,
    make_tiles,
    trim,
    join_stacks,
    csv_frame,
)
from .plates import add_global_xy, add_row_col
from .pool_design import reverse_complement as rc
from . import in_situ
