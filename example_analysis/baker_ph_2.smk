# Import necessary modules and libraries
import os
import snakemake
import ops.firesnake
from ops.firesnake import Snake
from ops.imports import *
import ops.constants
import ops.io
from ops.process import find_foci

# Reduce BLAS-induced thread oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Reduce numpy overhead
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Set parameters for image processing
SMOOTH = 5
RADIUS = 30
THRESHOLD_DAPI = 2500
NUCLEUS_AREA = (250, 3500)

# Set channel information
BACKGROUND_CHANNEL = 2  # Membrane channel
FOCI_CHANNEL = 1  # Cenpa channel

# Define threshold values
BACKGROUND_THRESHOLD_OFFSET = 80
BACKGROUND_QUANTILE = {'low': 0.05, 'mid': 0.025, 'high': 0.01}
CELL_COUNT_THRESHOLDS = (1400, 2500)
BACKGROUND_EROSION = 50
BACKGROUND_SMOOTH = 3

# Define plate layout
WELLS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']   # List of well identifiers
TILES = list(range(1281))  # List of tile numbers + 1

# Channel information
FINAL_CHANNELS = 'DAPI-GFP-A594-AF750'
CHANNEL_NAMES=['dapi','cenpa','coxiv','wga']

# Define colormap
GLASBEY_INVERTED = (ops.constants.GLASBEY_INVERTED)

# Define pattern for preprocessing image files
preprocess_pattern = 'input_ph/process/input/20X/multidimensional/20X_{{well}}_Tile-{{tile}}.phenotype.tif'

# Defines the final output files for the pipeline, ensuring generation of files for each combination of well and tile
rule all:
    input:
        # Each file is generated for each combination of well and tile
        expand('process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv', well=WELLS, tile=TILES),
        expand('process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv',well=WELLS,tile=TILES)

# Rule to correct align images from each well and tile in the phenotype channel
rule correct_align:
    input:
        expand(preprocess_pattern, well=WELLS, tile=TILES)
    output:
        'process_ph/images/20X_{{well}}_{channels}_Tile-{{tile}}.aligned.tif'.format(channels=FINAL_CHANNELS)
    run:
        # Print the list of input files
        print(input)
        # Read the first input file to check its shape
        data = read(input[0])
        # Print the shape of the read data
        print(data.shape)
        # Call the Snake align_phenotype_channels method with specified parameters
        Snake.align_phenotype_channels(
            # Output file path for aligned images
            output=output,
            # Image data to align, assuming input is grayscale (single channel)
            data=data,
            # Target channel index for alignment (dapi)
            target=0,
            # Source channel index for alignment
            source=2,
            # Riders channel index for alignment
            riders=[]
        )

# Rule to segment nuclei from aligned phenotype channel images
rule segment_nuclei:
    input:
        'process_ph/images/20X_{{well}}_{channels}_Tile-{{tile}}.aligned.tif'.format(channels=FINAL_CHANNELS)
    output:
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    run:
        # Call the Snake segment_nuclei method with specified parameters
        Snake.segment_nuclei(
            # Output file path for segmented nuclei images
            output=output,
            # Input data file path, assumed to be aligned phenotype channel images
            data=input[0],
            # Threshold value for segmenting nuclei based on DAPI signal
            threshold=THRESHOLD_DAPI,
            # Minimum and maximum area values for nuclei
            area_min=NUCLEUS_AREA[0],
            area_max=NUCLEUS_AREA[1],
            # Smoothing factor for preprocessing
            smooth=SMOOTH,
            # Radius value for preprocessing
            radius=RADIUS,
            # Lookup tables for visualization
            luts=GLASBEY_INVERTED
        )


# Rule to segment cells from aligned phenotype channel images and segmented nuclei images
rule segment_cells:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.tif'.format(channel=FINAL_CHANNELS),
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_ph/images/20X_{well}_Tile-{tile}.cells.tif'
    run:
        # Read the segmented nuclei images
        nuclei = ops.io.read_stack(input[1])

        # Determine quantile based on the maximum value in the segmented nuclei images
        if nuclei.max() < CELL_COUNT_THRESHOLDS[0]:
            quantile = BACKGROUND_QUANTILE['low']
        elif nuclei.max() > CELL_COUNT_THRESHOLDS[1]:
            quantile = BACKGROUND_QUANTILE['high']
        else:
            quantile = BACKGROUND_QUANTILE['mid']

        # Call the Snake segment_cells_robust method with specified parameters
        Snake.segment_cells_robust(
            # Output file path for segmented cells images
            output=output,
            # Input data file path, assumed to be aligned phenotype channel images
            data=input[0],
            # Channel index for background signal
            channel=BACKGROUND_CHANNEL,
            # Segmented nuclei images
            nuclei=nuclei,
            # Offset for background threshold
            background_offset=BACKGROUND_THRESHOLD_OFFSET,
            # Background quantile for thresholding
            background_quantile=quantile,
            # Smoothing factor for preprocessing
            smooth=BACKGROUND_SMOOTH,
            # Erosion factor for preprocessing
            erosion=BACKGROUND_EROSION,
            # Compression factor for output
            compress=1
        )

# Rule to extract phenotype information from segmented nuclei images
rule phenotype_info:
    input: 
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv'
    run:
        # Call the Snake extract_phenotype_minimal method with specified parameters
        Snake.extract_phenotype_minimal(
            # Output file path for phenotype information CSV file
            output=output, 
            # Segmented phenotype channel images (nuclei images)
            data_phenotype=input[0], 
            # Segmented nuclei images
            nuclei=input[0], 
            # Wildcards for capturing dynamic parts of the file paths
            wildcards=wildcards
        )

# Rule to extract phenotype information using CellProfiler from phenotype images using nuclei and cells
rule extract_phenotype_cp:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.tif'.format(channel=FINAL_CHANNELS),
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif',
        'process_ph/images/20X_{well}_Tile-{tile}.cells.tif'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv'
    benchmark:
        'process_ph/benchmark/20X_{well}_Tile-{tile}.benchmark_cp_phenotype.tsv'
    run:
        # Call the Snake extract_phenotype_cp_multichannel method with specified parameters
        Snake.extract_phenotype_cp_multichannel(
            # Output file path for phenotype information CSV file
            output=output,
            # Segmented phenotype channel images
            data_phenotype=input[0],
            # Segmented nuclei images
            nuclei=input[1],
            # Segmented cells images
            cells=input[2],
            # Wildcards for capturing dynamic parts of the file paths
            wildcards=wildcards,
            # Channel index for foci channel
            foci_channel=FOCI_CHANNEL,
            # Ordered set of channel names
            channel_names=CHANNEL_NAMES,
            # Maximum number of workers to use
            maxworkers=1
        )
