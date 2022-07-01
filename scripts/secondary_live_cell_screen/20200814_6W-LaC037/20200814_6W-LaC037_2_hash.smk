import snakemake
import ops.firesnake
from ops.firesnake import Snake

threshold_point = 8
det_range = (0.06,0.065)
initial_sites = [(0, 0),
 (19, 10),
 (210, 60),
 (399, 110),
 (380, 120),
 (190, 60),
 (200, 65),
 (180, 55)]

# WELLS,TILES,SITES = df_align[['well','tile','site']].values.T

rule all:
    input:
        # request individual files or list of files
        'fast_alignment.hdf'

rule fast_alignment:
    input:
        'ph_metadata_final_frame.pkl',
        'tracked_last_frame.hdf',
        'input_sbs/metadata/10X_c1_A2_DAPI-CY3-A594-CY5-CY7.metadata.pkl',
        'sbs_info.hdf'
    output:
        'fast_alignment.hdf'
    threads: 96
    run:
        import ops.utils
        import ops.triangle_hash as th
        from joblib import Parallel,delayed
        import pandas as pd

        f_ph_metadata = input[0]
        f_ph_info = input[1]
        f_sbs_metadata = input[2]
        f_sbs_info = input[3]

        df_ph_info = pd.read_hdf(f_ph_info)
        df_sbs_info = pd.read_hdf(f_sbs_info).rename(columns={'i':'j','j':'i'})

        df_ph_info_hash = (df_ph_info
            .pipe(ops.utils.gb_apply_parallel,['tile'],th.find_triangles,n_jobs=threads,tqdm=False)
            )
        df_sbs_info_hash = (df_sbs_info
            .pipe(ops.utils.gb_apply_parallel,['tile'],th.find_triangles,n_jobs=threads,tqdm=False)
            .rename(columns={'tile':'site'})
            )

        df_ph_xy = (pd.read_pickle(f_ph_metadata)
            .rename(columns={'site':'tile','x_data':'x','y_data':'y'})
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
            threshold_point=threshold_point,
            tqdm=False,
            n_jobs=threads
            )

        df_align.to_hdf(output[0],'x',mode='w')