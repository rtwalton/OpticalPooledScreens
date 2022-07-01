import snakemake
import ops.firesnake
from ops.firesnake import Snake
# import ops.constants
from ops.timelapse import call_TrackMate_centroids
from ops.cp_emulator import grayscale_features, shape_features, intensity_columns, shape_columns

SMOOTH = 10
RADIUS = 60
THRESHOLD_mCHERRY = 550
NUCLEUS_AREA = (500,10000)

WELLS = ['A2']
TILES = list(range(400))
FINAL_CHANNELS = 'mCherry'

DRIFT_FRAMES = 7
# FIJI = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
FIJI = '~/Fiji.app/ImageJ-linux64'
TRACKER_SETTINGS = {"LINKING_MAX_DISTANCE":60.,"GAP_CLOSING_MAX_DISTANCE":60.,
                        "ALLOW_TRACK_SPLITTING":True,"SPLITTING_MAX_DISTANCE":60.,
                        "ALLOW_TRACK_MERGING":True,"MERGING_MAX_DISTANCE":60.,
                        "MAX_FRAME_GAP":2,"CUTOFF_PERCENTILE":0.9}

features = {**grayscale_features,**shape_features}
feature_columns = {**intensity_columns,**shape_columns}

_ = features.pop('intensity_distribution')
_ = features.pop('zernike')

print('Will calculate {} phenotype features per labeled object'.format(len(features)))

# .tif file metadata recognized by ImageJ
# DISPLAY_RANGES = ((500, 20000))

rule all:
    input:
        # request individual files or list of files
        expand('process_ph/tables/20X_{well}_Tile-{tile}.tracked.csv',well=WELLS,tile=TILES),
        expand('process_ph/images/20X_{well}_Tile-{tile}.nuclei_tracked.tif',well=WELLS,tile=TILES),
        expand('process_ph/tables/20X_{well}_Tile-{tile}.phenotype.csv',well=WELLS,tile=TILES)

rule apply_illumination_correction:
    input:
        'input_ph/preprocess/timelapse/20X_{well}_{channel}_Site-{tile}.tif',
        'input_ph/illumination_correction/20X_{well}_{channel}.illumination_correction.tif'
    output:
        'process_ph/images/20X_{well}_{channel}_Tile-{tile}.aligned.hdf'
    benchmark: 'process_ph/benchmark/20X_{well}_{channel}_Tile-{tile}.aligned-benchmark.csv'
    threads: 2
    run:
        corrected = Snake.apply_illumination_correction(data=input[0],correction=input[1],n_jobs=threads)
        Snake.align_stage_drift(output=output,data=corrected,frames=DRIFT_FRAMES)

rule segment_nuclei:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS)
    output:
        temp('process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif')
    threads: 4
    benchmark: 'process_ph/benchmark/20X_{well}_mCherry_Tile-{tile}.segment-benchmark.csv'
    run:
        Snake.segment_nuclei_stack(output=output,data=input[0], threshold=THRESHOLD_mCHERRY,
            area_min=NUCLEUS_AREA[0], area_max=NUCLEUS_AREA[1],
            smooth=SMOOTH, radius=RADIUS, n_jobs=threads, compress=1)

rule trackmate_input:
    input:
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif'
    output:
        temp('process_ph/tables/20X_{well}_Tile-{tile}.nuclei_coords.csv')
    benchmark: 'process_ph/benchmark/20X_{well}_Tile-{tile}.trackmate-input-benchmark.csv'
    run:
        Snake.extract_timelapse_features(output=output,data=input[0],
            labels=input[0],wildcards=wildcards)

rule run_trackmate:
    input:
        'process_ph/tables/20X_{well}_Tile-{tile}.nuclei_coords.csv'
    output:
        temp('process_ph/tables/20X_{well}_Tile-{tile}.trackmate.csv')
    benchmark: 'process_ph/benchmark/20X_{well}_Tile-{tile}.trackmate-benchmark.csv'
    threads: 1
    run:
        call_TrackMate_centroids(input_path=input[0],output_path=output[0],
            fiji_path=FIJI,threads=threads,tracker_settings=TRACKER_SETTINGS)

rule relabel_tracked:
    input:
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif',
        'process_ph/tables/20X_{well}_Tile-{tile}.trackmate.csv',
        'process_ph/tables/20X_{well}_Tile-{tile}.nuclei_coords.csv'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.tracked.csv',
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei_tracked.tif'
    benchmark: 'process_ph/benchmark/20X_{well}_Tile-{tile}.relabel-benchmark.csv'
    run:
        Snake.relabel_trackmate(output=output,nuclei=input[0],
            df_trackmate=input[1],df_nuclei_coords=input[2], compress=1)

rule extract_timelapse_features:
    input:
        'process_ph/images/20X_{{well}}_{channel}_Tile-{{tile}}.aligned.hdf'.format(channel=FINAL_CHANNELS),
        'process_ph/images/20X_{well}_Tile-{tile}.nuclei_tracked.tif'
    output:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype.csv'
    benchmark: 'process_ph/benchmark/20X_{well}_Tile-{tile}.phenotype-benchmark.csv'
    run:
        df_ph = Snake.extract_timelapse_features(data=input[0],labels=input[1],wildcards=wildcards,features=features,maxworkers=4)
        df_ph = df_ph.rename(columns=feature_columns)
        df_ph.to_csv(output[0],index=None)

