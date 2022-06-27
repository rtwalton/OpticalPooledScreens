import snakemake
import ops.firesnake
from ops.firesnake import Snake

ROWS = ['A','B']
COLUMNS = list(range(1,4))
WELLS = [row+str(column) for row in ROWS for column in COLUMNS]
TILES_PH = list(range(1281))
SITES_SBS = list(range(333))

PH_CYCLE = 'DAPI-GFP'
PH_CHANNELS = 'GFP-DAPI'

SBS_CYCLE = 'c1'
SBS_CHANNELS = 'DAPI-CY3-A594-CY5-CY7'

det_range = (0.06,0.065)
initial_sites = [(1,1),(186,50),(548,150),(656,174),(887,225),(1279,331)]

# WELLS,TILES,SITES = df_align[['well','tile','site']].values.T

rule all:
    input:
        # request individual files or list of files
        expand('ph_info_{well}.hdf',well=WELLS),
        expand('sbs_info_{well}.hdf',well=WELLS),
        'fast_alignment_all.hdf'

rule fast_alignment:
    input:
        'input_ph/metadata/20X_{cycle}_{{well}}_{channel}.metadata.pkl'.format(cycle=PH_CYCLE,channel=PH_CHANNELS),
        expand('process_ph/tables/20X_{{well}}_Tile-{tile}.phenotype_info.csv',tile=TILES_PH),
        'input_sbs/metadata/10X_{cycle}_{{well}}_{channel}.metadata.pkl'.format(cycle=SBS_CYCLE,channel=SBS_CHANNELS),
        expand('process_sbs/tables/10X_{{well}}_Tile-{tile}.sbs_info.csv',tile=SITES_SBS)
    output:
        'ph_info_{well}.hdf',
        'sbs_info_{well}.hdf',
        'fast_alignment_{well}.hdf'
    threads: 96
    run:
        import ops.utils
        import ops.triangle_hash as th
        from joblib import Parallel,delayed
        import pandas as pd

        f_ph_metadata = input[0]
        f_ph_info = input[1:len(TILES_PH)+1]
        f_sbs_metadata = input[len(TILES_PH)+1]
        f_sbs_info = input[(len(TILES_PH)+2):]

        def get_file(f):
            try:
                return pd.read_csv(f)
            except pd.errors.EmptyDataError:
                pass

        arr_ph = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in f_ph_info)
        df_ph_info = pd.concat(arr_ph)

        df_ph_info.to_hdf(output[0],'x',mode='w')

        arr_sbs = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in f_sbs_info)
        df_sbs_info = pd.concat(arr_sbs)

        df_sbs_info.to_hdf(output[1],'x',mode='w')

        df_ph_info_hash = (df_ph_info
            .pipe(ops.utils.gb_apply_parallel,['tile'],th.find_triangles,n_jobs=threads,tqdm=False)
            )
        df_sbs_info_hash = (df_sbs_info
            .pipe(ops.utils.gb_apply_parallel,['tile'],th.find_triangles,n_jobs=threads,tqdm=False)
            .rename(columns={'tile':'site'})
            )

        df_ph_xy = (pd.read_pickle(f_ph_metadata)
            .rename(columns={'field_of_view':'tile'})
            .set_index('tile')
            [['x','y']]
            )

        df_sbs_xy = (pd.read_pickle(f_sbs_metadata)
            .rename(columns={'field_of_view':'tile'})
            .set_index('tile')
            [['x', 'y']]
           )

        df_align = th.multistep_alignment(
            df_ph_info_hash,
            df_sbs_info_hash,
            df_ph_xy,
            df_sbs_xy,
            det_range=det_range,
            initial_sites=initial_sites,
            tqdm=False,
            n_jobs=threads
            )

        df_align.assign(well=wildcards.well).to_hdf(output[2],'x',mode='w')

rule combine_alignment:
    input:
        expand('fast_alignment_{well}.hdf',well=WELLS)
    output:
        'fast_alignment_all.hdf'
    run:
        import pandas as pd
        df_alignment = pd.concat([pd.read_hdf(f).assign(well=f[-6:-4]) for f in input])
        df_alignment.to_hdf(output[0],'x',mode='w')
