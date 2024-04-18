# Import necessary modules and libraries
import snakemake
import ops.firesnake
from ops.firesnake import Snake
from ops.imports import *
import pandas as pd
import os

# Define lists of cycles
SBS_CYCLES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# A string representing bases
BASES = 'GTAC' 

# Define threshold values
THRESHOLD_READS = 315  # Threshold for detecting reads
THRESHOLD_DAPI = 1200  # Threshold for segmenting nuclei based on dapi signal
THRESHOLD_CELL = {'A1': 750,'A2': 750,'A3': 750,'B1': 750,'B2': 750,'B3': 750}  # Threshold for segmenting cells on sbs background signal, specific to well 'A1'
NUCLEUS_AREA = (0.25 * 150, 0.25 * 800)  # Tuple representing nucleus area
Q_MIN = 0  # Minimum Levenshtein distance

# Define plate layout
WELLS = ['A1','A2','A3','B1','B2','B3']  # List of well identifiers
TILES = list(range(333))  # List of tile numbers + 1

# Read data from a CSV file named 'barcodes.csv' and store it in a DataFrame
df_pool = pd.read_csv('sbs_1/barcodes.csv', index_col=None)

# Define a pattern for preprocessing image files
preprocess_pattern = 'input_sbs/process/input/10X/multidimensional/10X_c{cycle}-SBS-{cycle}_{{well}}_Tile-{{tile}}.sbs.tif'

# Define display ranges for different channels, recognized by ImageJ
DISPLAY_RANGES = [
    [500, 15000],  # Range for DAPI channel
    [100, 10000],  # Range for CY3 channel
    [100, 10000],  # Range for A594 channel
    [200, 25000],  # Range for CY5 channel
    [200, 25000]   # Range for CY7 channel
]

# Define LUTs (Lookup Tables) for different channels
LUTS = [
    ops.io.GRAY,    # Lookup table for DAPI channel
    ops.io.GREEN,   # Lookup table for CY3 channel
    ops.io.RED,     # Lookup table for A594 channel
    ops.io.MAGENTA, # Lookup table for CY5 channel
    ops.io.CYAN     # Lookup table for CY7 channel
]

# Defines the final output files for the pipeline, ensuring generation of files for each combination of well and tile
rule all:
    input:
        # Each file is generated for each combination of well and tile
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv', well=WELLS, tile=TILES),
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv', well=WELLS, tile=TILES)

# Aligns images from each sequencing round 
rule align:
    priority: -1
    input:
        # List comprehension to generate input file paths for each cycle
        ([[preprocess_pattern.format(cycle=SBS_CYCLES[cycle])] for cycle in range(0,len(SBS_CYCLES))])
    output:
        # Output file path for aligned images
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif'
    run:
        # Read input data, align using specified method, and print the shape
        data = np.array([read(f) for f in input])
        print(data.shape)
        # Call Snake align_SBS method with specified parameters
        Snake.align_SBS(
            # Output file path for aligned images
            output=output, 
            # Image data to align, expected dimensions of (CYCLE, CHANNEL, I, J)
            data=data, 
            # Method for aligning images across cycles
            method='SBS_mean', 
            # Display ranges for different channels, recognized by ImageJ
            display_ranges=DISPLAY_RANGES, 
            # Lookup Tables for different channels
            luts=LUTS
        )

# Applies Laplacian-of-Gaussian filter to all channels
rule transform_LoG:
    priority: -1
    input:
        # Input file path for aligned images
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif'
    output:
        # Output file path for Laplacian-of-Gaussian transformed images
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    run:
        # Read input data and print the shape
        data = read(input[0])
        print(data.shape)
        # Call Snake transform_log method with specified parameters
        Snake.transform_log(
            # Output file path for Laplacian-of-Gaussian transformed images
            output=output, 
            # Aligned SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J)
            data=input, 
            # Skip transforming a specific channel (e.g., DAPI with skip_index=0)
            skip_index=0,
            # Display ranges for different channels, recognized by ImageJ
            display_ranges=DISPLAY_RANGES, 
            # Lookup Tables for different channels
            luts=LUTS
        )

# Computes standard deviation of SBS reads across cycles
rule compute_std:
    input:
        # Input file path for the LoG-ed SBS image
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        # Output file path for the computed standard deviation image
        'process_sbs/images/10X_{well}_Tile-{tile}.std.tif'
    run:
        # Read the input LoG-ed SBS image data
        data = read(input[0])
        # Print the shape of the input data
        print(data.shape)
        # Call the Snake compute_std method with specified parameters
        Snake.compute_std(
            # Output file path for the computed standard deviation image
            output=output, 
            # LoG-ed SBS image data with dimensions (CYCLE, CHANNEL, I, J)
            data=input[0], 
            # Index of data to remove from subsequent analysis, generally non-SBS channels
            remove_index=0
        )

# Find local maxima of SBS reads across cycles
rule find_peaks:
    input:
        # Input file path for the standard deviation image
        'process_sbs/images/10X_{well}_Tile-{tile}.std.tif'
    output:
        # Output file path for the peaks image
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif'
    run:
        # Read the input standard deviation image data
        data = read(input[0])
        # Print the shape of the input data
        print(data.shape)
        # Call the Snake find_peaks method with specified parameters
        Snake.find_peaks(
            # Output file path for the peaks image
            output=output, 
            # Standard deviation image data
            data=input[0]
        )

# Dilates sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    input:
        # Input file path for the Laplacian-of-Gaussian filtered SBS data
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        # Output file path for the maxed data
        'process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif'
    run:
        # Read the input data
        data = read(input[0])
        # Print the shape of the input data
        print(data.shape)
        # Call the Snake max_filter method with specified parameters
        Snake.max_filter(
            # Output file path for the maxed data
            output=output,
            # Input data path
            data=input[0],
            # Neighborhood size for max filtering
            width=3,
            # Index of data to remove from subsequent analysis
            remove_index=0
        )

# Segment nuclei from the DAPI channel
rule segment_nuclei:
    input:
        # Input file path for the aligned image data
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif'
    output:
        # Output file path for the segmented nuclei
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    run:
        # Read the input data
        data = read(input[0])
        # Print the shape of the input data
        print(data.shape)
        # Call the Snake segment_nuclei method with specified parameters
        Snake.segment_nuclei(
            # Output file path for the segmented nuclei
            output=output, 
            # Image data with the DAPI channel
            data=data[0], 
            # Threshold for mean DAPI intensity
            threshold=THRESHOLD_DAPI,
            # Minimum area for retaining nuclei
            area_min=NUCLEUS_AREA[0], 
            # Maximum area for retaining nuclei
            area_max=NUCLEUS_AREA[1]
        )

# Segment cells from aligned data
rule segment_cells:
    input:
        # Input file paths for the aligned image data and segmented nuclei
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        # Output file path for the segmented cells
        'process_sbs/images/10X_{well}_Tile-{tile}.cells.tif'
    run:
        # Read the input data
        data = read(input[0])
        # Print the shape of the input data
        print(data.shape)
        # Call the Snake segment_cells method with specified parameters
        Snake.segment_cells(
            # Output file path for the segmented cells
            output=output, 
            # Image data for cell boundary segmentation
            data=data[0], 
            # Labeled segmentation mask of nuclei
            nuclei=input[1], 
            # Threshold used to identify cell boundaries
            threshold=THRESHOLD_CELL[wildcards.well]
        )

# Extract bases from peaks
rule extract_bases:
    input:
        # Input file paths for peaks, maxed, and cells images
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.cells.tif'
    output:
        # Output file path for the bases CSV
        'process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv'
    run:
        # Call the Snake extract_bases method with specified parameters
        Snake.extract_bases(
            # Output file path for the bases CSV
            output=output, 
            # Peaks/local maxima score for each pixel
            peaks=input[0], 
            # Base intensity at each point
            maxed=input[1], 
            # Labeled segmentation mask of cell boundaries
            cells=input[2], 
            # Threshold for identifying candidate sequencing reads
            threshold_peaks=THRESHOLD_READS, 
            # Order of bases corresponding to the order of acquired SBS channels
            bases=BASES, 
            # Metadata to include in output table
            wildcards=dict(wildcards)
        )

# Call reads
rule call_reads:
    input:
        # Input file paths for bases CSV and peaks images
        'process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv',
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif'
    output:
        # Output file path for the reads CSV
        'process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv'
    run:
        # Call the Snake call_reads method with specified parameters
        Snake.call_reads(
            # Output file path for the reads CSV
            output=output, 
            # DataFrame containing base information
            df_bases=input[0], 
            # Array containing peak information
            peaks=input[1], 
            # Flag indicating if channel minimum should be subtracted from intensity
            subtract_channel_min=True
        )

# Call cells
rule call_cells:
    input:
        # Input CSV file containing read information
        'process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv'
    output:
        # Output CSV file containing corrected cells
        'process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv'
    run:
        # Call Snake call_cells method with specified parameters
        Snake.call_cells(
            # Output CSV file path for corrected cells
            output=output, 
            # DataFrame containing read information
            df_reads=input[0], 
            # DataFrame containing pool information, if available
            df_pool=df_pool, 
            # Minimum quality threshold
            q_min=Q_MIN
        )

# Extract minimal phenotype features
rule sbs_cell_info:
    input: 
        # Input file containing nuclei information
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        # Output CSV file containing extracted minimal phenotype features
        'process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv'
    run:
        # Call Snake extract_phenotype_minimal method with specified parameters
        Snake.extract_phenotype_minimal(
            # Output CSV file path for extracted minimal phenotype features
            output=output, 
            # DataFrame containing phenotype data
            data_phenotype=input[0], 
            # Array containing nuclei information
            nuclei=input[0], 
            # Metadata to include in output table
            wildcards=wildcards
        )