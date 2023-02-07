import os

# reduce BLAS-induced thread oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# reduce numpy overhead?
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION']='0'

import snakemake
import ops.firesnake
from ops.firesnake import Snake
# from ops.imports import *
import ops.constants
import ops.io

SMOOTH = 5
RADIUS = 45
THRESHOLD_DAPI = 550
NUCLEUS_AREA = (200,10000)

ACTIN_CHANNEL = 3
GH2AX_CHANNEL = 2

ACTIN_THRESHOLD_OFFSET = 65
ACTIN_BACKGROUND_QUANTILE = {'low':0.05,'mid':0.025,'high':0.01}
CELL_COUNT_THRESHOLDS = (1200,2100)
ACTIN_EROSION = 10
ACTIN_SMOOTH = 3

ROWS = ['A','B']
COLUMNS = list(range(1,4))
MISSING = ['B1']
WELLS = [row+str(column) for row in ROWS for column in COLUMNS]
WELLS = sorted(list(set(WELLS)-set(MISSING)))
TILES = list(range(1281))
CHANNELS = [['DAPI-GFP','A594','AF750']]
CYCLES = ['DAPI-GFP-A594-AF750']
FINAL_CHANNELS = 'DAPI-GFP-A594-AF750'
TILE_FILES=3

# .tif file metadata recognized by ImageJ
# DISPLAY_RANGES = ((500, 20000), 
#                 (800, 5000), 
#                 (800, 5000),
#                 (800, 5000),
#                 (800, 5000))

GLASBEY_INVERTED = (ops.constants.GLASBEY_INVERTED)

rule all:
    input:
        # request individual files or list of files
        expand('process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv', well=WELLS, tile=TILES),
        expand('process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv',well=WELLS,tile=TILES)

rule correct_align:
    input:
        ['input_ph/preprocess/{cycle}/20X_{cycle}_{{well}}_{channel}_Site-{{tile}}.tif'
            .format(cycle=cycle,channel=channel)
            for cycle_index,cycle in enumerate(CYCLES)
            for channel in CHANNELS[cycle_index] 
            ],
        ['input_ph/illumination_correction/20X_{cycle}_{{well}}_{channel}.illumination_correction.tif'
            .format(cycle=cycle,channel=channel)
            for cycle_index,cycle in enumerate(CYCLES)
            for channel in CHANNELS[cycle_index] 
            ]
    output:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS)
    run:
        corrected = [Snake.apply_illumination_correction(data=data,correction=correction,zproject=True) 
                            for data, correction
                            in zip(input[:TILE_FILES],input[TILE_FILES:])
                        ]
        stacked = Snake._stack_channels(data=corrected)
        Snake.align_phenotype_channels(output=output,data=stacked,target=0,source=2,riders=3)

rule segment_nuclei:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS)
    output:
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    run:
        Snake.segment_nuclei(output=output,data=input[0], threshold=THRESHOLD_DAPI,
            area_min=NUCLEUS_AREA[0], area_max=NUCLEUS_AREA[1],
            smooth=SMOOTH, radius=RADIUS, luts=GLASBEY_INVERTED)

rule phenotype_info:
    input: 
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv'
    run:
        Snake.extract_phenotype_minimal(output=output, 
            data_phenotype=input[0], nuclei=input[0], wildcards=wildcards)

rule segment_cells:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS),
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_ph/images/20X_{well}_Tile-{tile}.cells.tif'
    run:
        nuclei = ops.io.read_stack(input[1])

        if nuclei.max()<CELL_COUNT_THRESHOLDS[0]:
            quantile = ACTIN_BACKGROUND_QUANTILE['low']
        elif nuclei.max()>CELL_COUNT_THRESHOLDS[1]:
            quantile = ACTIN_BACKGROUND_QUANTILE['high']
        else:
            quantile = ACTIN_BACKGROUND_QUANTILE['mid']

        Snake.segment_cells_robust(output=output, data=input[0], channel=ACTIN_CHANNEL, nuclei=nuclei,
            background_offset=ACTIN_THRESHOLD_OFFSET, background_quantile=quantile,
            smooth=ACTIN_SMOOTH, erosion=ACTIN_EROSION, compress=1)

rule extract_phenotype_cp:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS),
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif',
        'process_ph/images/20X_{well}_Tile-{tile}.cells.tif'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv'
    benchmark:
        'process_ph/benchmark/20X_{well}_Tile-{tile}.benchmark_cp_phenotype.tsv'
    run:
        Snake.extract_phenotype_cp_multichannel(output=output, data_phenotype=input[0], nuclei=input[1], cells=input[2],
            wildcards=wildcards, foci_channel=GH2AX_CHANNEL, maxworkers=1)
