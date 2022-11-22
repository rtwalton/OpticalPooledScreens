import snakemake
import ops.firesnake
from ops.firesnake import Snake
from ops.imports import *
import pandas as pd

CYCLES = ['c'+str(num) for num in range(1,12)]
CYCLE_FILES = [1] + [4,]*(len(CYCLES)-1)
CHANNELS = ['DAPI-CY3-A594-CY5-CY7']+[['CY3','A594','CY5','CY7']]*(len(CYCLES)-1)

BASES = 'GTAC'

THRESHOLD_READS = 350  # threshold for detecting reads
THRESHOLD_DAPI = 500 # threshold for segmenting nuclei on dapi signal
THRESHOLD_CELL = 1700 # threshold for segmenting cells on sbs background signal
NUCLEUS_AREA = (20,800)
Q_MIN = 0 # levenshtein distance >= 2 for this library so not worried

ROWS = ['A','B']
COLUMNS = list(range(1,4))

# MISSING = ['B1']

WELLS = [row+str(column) for column in COLUMNS for row in ROWS]
# WELLS = list(set(WELLS)-set(MISSING))

TILES = list(range(333))

df_design = pd.read_csv('/luke-perm/libraries/pool10/pool10_design.csv',index_col=None)
df_pool = df_design.query('dialout==[0,1]').drop_duplicates('sgRNA')
df_pool['prefix'] = df_pool.apply(lambda x: x.sgRNA[:x.prefix_length],axis=1)

preprocess_pattern = 'input_sbs/preprocess/{cycle}/10X_{cycle}_{{well}}_{channel}_Site-{{tile}}.tif'

# .tif file metadata recognized by ImageJ
DISPLAY_RANGES = ((500, 20000), 
                (800, 5000), 
                (800, 5000),
                (800, 5000),
                (800, 5000))

LUTS = ops.io.GRAY, ops.io.GREEN, ops.io.RED, ops.io.MAGENTA, ops.io.CYAN

rule all:
    input:
        # request individual files or list of files
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv', well=WELLS, tile=TILES),
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv', well=WELLS, tile=TILES),
    
rule align:
    priority: -1
    input:
        ([[preprocess_pattern.format(cycle=CYCLES[0],channel=CHANNELS[0])]]+
            [[preprocess_pattern.format(cycle=CYCLES[cycle],channel=channel) for channel in CHANNELS[cycle]] for cycle in range(1,len(CYCLES))])
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif')
    run:
        Snake.align_SBS(output=output, data=input, method='SBS_mean', cycle_files=CYCLE_FILES, n=1,
            display_ranges=DISPLAY_RANGES, luts=LUTS, upsample_factor=1)

rule transform_LoG:
    priority: -1
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif'
    output:
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    run:
        Snake.transform_log(output=output, data=input, skip_index=0,
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule compute_std:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.std.tif')
    run:
        Snake.compute_std(output=output, data=input[0], remove_index=0)

rule find_peaks:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.std.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif')
    run:
        Snake.find_peaks(output=output, data=input[0]) 

rule max_filter:
    """Dilates sequencing channels to compensate for single-pixel alignment error.
    """
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif')
    run:
        Snake.max_filter(output=output,data=input[0],width=3,remove_index=0)
        # maxed = Snake._max_filter(data=read(input[0]), width=3,remove_index=0)
        # maxed[maxed==0]=1
        # save(str(output),maxed)

rule apply_illumination_correction:
    input:
        preprocess_pattern.format(cycle=CYCLES[0],channel=CHANNELS[0]),
        'input_sbs/illumination_correction/10X_c1_{{well}}_{channel}.illumination_correction.tif'.format(channel=CHANNELS[0])
    output:
        temp('process_sbs/images/10X_c1_{well}_Tile-{tile}.corrected.tif')
    run:
        Snake.apply_illumination_correction(output=output,data=input[0],correction=input[1])

rule segment_nuclei:
    input:
        'process_sbs/images/10X_c1_{well}_Tile-{tile}.corrected.tif',
        # discarded input, to change run order
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    run:
        Snake.segment_nuclei(output=output, data=input[0], 
            # threshold=THRESHOLD_DAPIS[wildcards['well']], 
            threshold=THRESHOLD_DAPI,
            area_min=NUCLEUS_AREA[0], 
            area_max=NUCLEUS_AREA[1])

rule segment_cells:
    input:
        'process_sbs/images/10X_c1_{well}_Tile-{tile}.corrected.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_sbs/images/10X_{well}_Tile-{tile}.cells.tif'
    run:
        Snake.segment_cells(output=output, 
            data=input[0], nuclei=input[1], 
            threshold=THRESHOLD_CELL)

rule extract_bases:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.cells.tif'
    output:
        'process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv'
    run:
        Snake.extract_bases(output=output, peaks=input[0], maxed=input[1], 
            cells=input[2], threshold_peaks=THRESHOLD_READS, 
            bases=BASES, wildcards=dict(wildcards)) 

rule call_reads:
    input:
        'process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv',
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif'
    output:
        'process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv'
    run:
        Snake.call_reads(output=output, df_bases=input[0], peaks=input[1], subtract_channel_min=True)

rule call_cells:
    input:
        'process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv'
    output:
        'process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv'
    run:
        Snake.call_cells(output=output, df_reads=input[0], df_pool=df_pool, q_min=Q_MIN)

rule sbs_cell_info:
    input: 
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv'
    run:
        Snake.extract_phenotype_minimal(output=output, 
            data_phenotype=input[0], nuclei=input[0], wildcards=wildcards)