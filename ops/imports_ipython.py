# Import all functions and classes from the 'ops.imports' module
from ops.imports import *

# Import the Snake class from the 'ops.firesnake' module
from ops.firesnake import Snake

# Import the Align class from the 'ops.process' module
from ops.process import Align

import IPython
# Enable automatic reloading of modules when code changes
IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
IPython.get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
# Add a progress_apply method to pandas GroupBy objects
tqdm.pandas()

from matplotlib.colors import ListedColormap
# Create a ListedColormap from the GLASBEY color map for use in plots
GLASBEY_PLT = ListedColormap((GLASBEY.reshape(3,256).T)/256)
