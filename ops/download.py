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
bia_files = f'{package_dir}/paper/BioImage_Archive.csv.gz'


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


def download_from_bioimage_archive(directory, plate='all', dataset='sequencing', 
        well='all', site='all', query=None, ascp=ascp_guess):
    """Download data from the BioImage Archive."""
    os.makedirs(directory,exist_ok=True)

    if not shutil.which(ascp):
        ascp = shutil.which('ascp')
        if ascp is None:
            print(f'Error: Aspera ascp executable not found at {ascp}')
            raise QuitError

    queries = ['(Dataset == @dataset)']
    if plate != 'all':
        queries.append('(Plate == @plate)')
    if well != 'all':
        queries.append('(Well == @well)')
    if site != 'all': 
        queries.append('(Site == @site)')
    if query is not None:
        queries.append(f'({query})')
  
    df_bia = (pd.read_csv(bia_files, low_memory=False)
     .query('&'.join(queries))
    )

    if df_bia.pipe(len)==0:
        raise ValueError('No valid images specified.')

    pairlist = f'{directory}/ascp_download_list.txt'
    write_pairlist(df_bia, pairlist)
    command = format_ascp_command(ascp, pairlist, local=directory)

    print(f'Downloading {len(df_bia)} files from the BioImage Archive with the command: {command}')
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error in downloading files using {ascp}. This can be a user network issue. '
            'Try a secure network with better connectivity.'
            )
        raise QuitError

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