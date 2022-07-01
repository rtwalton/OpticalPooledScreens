import snakemake
import ops.firesnake
from ops.firesnake import Snake
# import ops.io
# from ops.imports import *
# import ops.triangle_hash as th
import pandas as pd

gate = '0.060 <= determinant <= 0.065 & score > 0.1'

df_align = pd.read_hdf('fast_alignment_all.hdf').query(gate)

WELLS,TILES_PH,SITES_SBS = df_align[['well','tile','site']].values.T

rule all:
    input:
        # request individual files or list of files
        # expand('process/10X_{well}_Tile-{tile}.phenotype.csv', zip, well=WELLS, tile=TILES),
        # expand('process/10X_{well}_Tile-{tile}.cells.csv', zip, well=WELLS, tile=TILES),
        expand('alignment/{well}_Tile-{tile}_Site-{site}.merge.csv', zip, well=WELLS, tile=TILES_PH,site=SITES_SBS),
    
rule merge:
    input:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv',
        'process_sbs/tables/10X_{well}_Tile-{site}.sbs_info.csv'
    output:
        'alignment/{well}_Tile-{tile}_Site-{site}.merge.csv'
    run:
        Snake.merge_triangle_hash(output=output,df_0=input[0],df_1=input[1],
            alignment=df_align.query('well==@wildcards.well & tile==@wildcards.tile & site==@wildcards.site').iloc[0])
        # well,tile,site = wildcards.well, wildcards.tile, wildcards.site
        # alignment = df_align.query('well==@well & tile==@tile & site==@site').iloc[0]

        # df_0 = pd.read_csv(input[0])
        # df_1 = pd.read_csv(input[1]).rename(columns={'tile':'site'})

        # model = th.build_linear_model(alignment['rotation'],alignment['translation'])

        # th.merge_sbs_phenotype(df_0,df_1,model).to_csv(str(output))
