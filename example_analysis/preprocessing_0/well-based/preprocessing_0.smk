import ops
import os
import glob
import tifffile
import numpy as np
import ops.filenames
from ops.preprocessing_smk import *
from ops.process import calculate_illumination_correction
from ops.io import save_stack as save

# Define wells and SBS (Sequencing By Synthesis) cycles
WELLS = ['A1','A2', 'A3', 'B1', 'B2', 'B3'] 
SBS_CYCLES = list(range(1, 12))

# Define channels for PH (Phenotyping) and SBS (Sequencing By Synthesis) images
PH_CHANNELS = ['DAPI_1x1-GFP_1x1', 'A594_1x1', 'AF750_1x1']  
SBS_CHANNELS = ['DAPI-CY3_30p_545-A594_30p-CY5_30p-CY7_30p', 'CY3_30p_545', 'A594_30p', 'CY5_30p', 'CY7_30p']

# Define tiles for PH (Phenotyping) and SBS (Sequencing By Synthesis) images
SBS_TILES = list(range(333))
PH_TILES = list(range(1281))

# File patterns for SBS and PH images
SBS_INPUT_PATTERN = '/lab/barcheese01/screens/aconcagua/input_sbs/c{cycle}/**/Well{well}*Channel{channel}_Seq*.nd2'
PH_INPUT_PATTERN = '/lab/barcheese01/screens/aconcagua/input_ph/**/**/Well{well}*Channel{channel}_Seq*.nd2'

# Parse function parameters
parse_function_home = "/lab/barcheese01/screens"
parse_function_dataset = "aconcagua"

# Final output files
rule all:
    input:
        expand("metadata/10X_c{cycle}-SBS-{cycle}_{well}.metadata.pkl", well=WELLS, cycle=SBS_CYCLES),
        expand("metadata/20X_{well}.metadata.pkl", well=WELLS),
        expand("input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.Channel-{channel}.tif", cycle=SBS_CYCLES[0],channel=SBS_CHANNELS[0],well=WELLS,tile=SBS_TILES),
        expand("input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.Channel-{channel}.tif", cycle=SBS_CYCLES[1:],channel=SBS_CHANNELS[1:],well=WELLS,tile=SBS_TILES),
        expand("input_ph_tif/20X_{well}_Tile-{tile}.phenotype.Channel-{channel}.tif", channel=PH_CHANNELS,well=WELLS,tile=PH_TILES),
        expand("illumination_correction/10X_c1-SBS-1_{well}.sbs.illumination_correction.tif", well=WELLS),
        expand("illumination_correction/20X_{well}.phenotype.Channel-{channel}.illumination_correction.tif", well=WELLS, channel=PH_CHANNELS),

# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        lambda wildcards: glob.glob(SBS_INPUT_PATTERN.format(cycle=wildcards.cycle, well=wildcards.well, channel="*"))[0]
    output:
        "metadata/10X_c{cycle}-SBS-{cycle}_{well}.metadata.pkl"
    run:
        os.makedirs('metadata', exist_ok=True)
        Snake_preprocessing.extract_metadata_well(
            output=output,
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset
        )

# Extract metadata for PH images
rule extract_metadata_ph:
    input:
        lambda wildcards: glob.glob(PH_INPUT_PATTERN.format(well=wildcards.well, channel="*"))[0]
    output:
        "metadata/20X_{well}.metadata.pkl"
    run:
        os.makedirs('metadata', exist_ok=True)
        Snake_preprocessing.extract_metadata_well(
            output=output,
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset
        )

# Convert SBS ND2 images to TIFF
rule convert_sbs:
    input:
        lambda wildcards: glob.glob(SBS_INPUT_PATTERN.format(cycle=wildcards.cycle, well=wildcards.well, channel=wildcards.channel))
    output:
        expand("input_sbs_tif/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}_Tile-{tile}.sbs.Channel-{{channel}}.tif", tile=SBS_TILES)
    run:
        os.makedirs('input_sbs_tif', exist_ok=True)
        results = Snake_preprocessing.convert_to_tif_well(
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            channel_order_flip=False,
            separate_dapi=False,
        )
        for fov_index, (image_array, fov_description) in results.items():
            output_filename = ops.filenames.name_file_channel(fov_description)
            print(output_filename)
            save(output_filename, image_array)
            
# Convert PH ND2 images to TIFF
rule convert_ph:
    input:
        lambda wildcards: glob.glob(PH_INPUT_PATTERN.format(well=wildcards.well, channel=wildcards.channel))
    output:
        expand("input_ph_tif/20X_{{well}}_Tile-{tile}.phenotype.Channel-{{channel}}.tif", tile=PH_TILES)
    resources:
        mem_mb=96000      
    run:
        os.makedirs('input_ph_tif', exist_ok=True)
        results = Snake_preprocessing.convert_to_tif_well(
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            channel_order_flip=False,
            separate_dapi=False,
        )
        for fov_index, (image_array, fov_description) in results.items():
            output_filename = ops.filenames.name_file_channel(fov_description)
            print(output_filename)
            save(output_filename, image_array)

# Calculate illumination correction for sbs files
rule calculate_icf_sbs:
    input:
        lambda wildcards: expand("input_sbs_tif/10X_c1-SBS-1_{well}_Tile-{tile}.sbs.Channel-DAPI-CY3_30p_545-A594_30p-CY5_30p-CY7_30p.tif", well=wildcards.well, tile=SBS_TILES)
    output:
        "illumination_correction/10X_c1-SBS-1_{well}.sbs.illumination_correction.tif"
    resources:
        mem_mb=96000
    run:
        os.makedirs('illumination_correction', exist_ok=True)
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)

# Calculate illumination correction for ph files
rule calculate_icf_ph:
    input:
        lambda wildcards: expand("input_ph_tif/20X_{well}_Tile-{tile}.phenotype.Channel-{{channel}}.tif", well=wildcards.well, channel=wildcards.channel, tile=PH_TILES)
    output:
        "illumination_correction/20X_{well}.phenotype.Channel-{channel}.illumination_correction.tif"
    resources:
        mem_mb=96000
    run:
        os.makedirs('illumination_correction', exist_ok=True)
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)