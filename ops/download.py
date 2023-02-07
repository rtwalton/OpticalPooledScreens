import fire
import os
import sys
import subprocess
import shutil

import pandas as pd

package_dir = os.path.sep.join(
    os.path.normpath(__file__).split(os.path.sep)[:-2])
aspera_openssh = os.path.join(package_dir, 'paper/asperaweb_id_dsa.openssh')
ascp_guess = os.path.join(os.environ['HOME'], '.aspera/connect/bin/ascp')
cell_idr_files = f'{package_dir}/paper/BioImage_Archive.csv.gz'


def write_pairlist(df_files, filename, remote_prefix='/fire/S-BIAD/394/S-BIAD394/Files/'):
    """Write list of filenames for use with ascp --file-pair-list option.
    """
    txt = []
    for f in df_files['path']:
        txt += [remote_prefix + f, f]
    txt = '\n'.join(txt)

    with open(filename, 'w') as fh:
        fh.write(txt)


def format_ascp_command(ascp, pairlist, local='.'):
    """Generate Aspera download command. Requires paths to ascp executable and SSH key. 
    See https://idr.openmicroscopy.org/about/download.html
    """
    ascp_opts = f'-T -l200m -P 33001 -i {aspera_openssh}'
    pair_opts = f'--file-pair-list={pairlist} --mode=recv --user=bsaspera --host=fasp-beta.ebi.ac.uk'
    return f'{ascp} {ascp_opts} {pair_opts} {local}'


def download_from_bioimage_archive(directory, query='', ascp=ascp_guess):
    """Download data from Cell IDR."""
    os.makedirs(directory,exist_ok=True)

    if not shutil.which(ascp):
        ascp = shutil.which('ascp')
        if ascp is None:
            print(f'Error: Aspera ascp executable not found at {ascp}')
            raise QuitError

    # # select our example
    # select_tile = f'idr_name == "experiment{experiment}"'
    # if well != 'all':
    #     select_tile += ' & well == @well'
    # if tile != 'all': 
    #     select_tile += ' & tile == @tile'

    # select_image_tags = 'tag == ["phenotype", "sbs"]'    
    df_idr = (pd.read_csv(cell_idr_files, low_memory=False)
     .query(query)
    #  .query(select_image_tags)
    )

    if df_idr.pipe(len)==0:
        raise ValueError('No valid tiles specified for the chosen experiment.')

    pairlist = f'{directory}/ascp_download_list.txt'
    write_pairlist(df_idr, pairlist)
    command = format_ascp_command(ascp, pairlist, local=directory)

    print(f'Downloading {len(df_idr)} files from Cell-IDR with command: {command}')
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error in downloading files using {ascp}. This can be a user network issue. '
            'Try a secure network with better connectivity.'
            )
        raise QuitError

    # well_tile_list = f'{directory}/experiment{experiment}/well_tile_list_example.csv'
    # df_idr[['well','tile']].drop_duplicates().to_csv(well_tile_list, index=None)

class QuitError(Exception):
    """Don't generate a stack trace if encountered in command line app.
    """
    pass


if __name__ == '__main__':
    commands = {
        'download_from_bioimage_archive': download_from_bioimage_archive,
    }
    try:
        fire.Fire(commands)
    except QuitError:
        sys.exit(1)