"""
BioImage Archive Data Download Utilities

This module provides utilities for downloading data from the BioImage Archive using Aspera's `ascp` tool.
It includes functions for:

1. Writing File Lists: Generating a file pair list for Aspera downloads.
2. Formatting Aspera Commands: Creating download commands for Aspera with specified options.
3. Downloading Data: Managing the download process from the BioImage Archive, including filtering by dataset, plate, well, and site.

"""

import fire
import os
import sys
import subprocess
import shutil
import pandas as pd

# Define paths for Aspera SSH key and executable, and BioImage Archive metadata file
package_dir = os.path.sep.join(
    os.path.normpath(__file__).split(os.path.sep)[:-2])
aspera_openssh = os.path.join(package_dir, 'paper/asperaweb_id_dsa.openssh')
ascp_guess = os.path.join(os.environ['HOME'], '.aspera/connect/bin/ascp')
bia_files = f'{package_dir}/paper/BioImage_Archive.csv.gz'

def write_pairlist(df_files, filename, remote_prefix='/fire/S-BIAD/394/S-BIAD394/Files/'):
    """Write a list of filenames to a text file for use with Aspera's --file-pair-list option.

    Args:
        df_files (DataFrame): DataFrame containing the file paths to be listed.
        filename (str): Path to the output text file that will contain the file pairs.
        remote_prefix (str): Prefix for the remote file paths.
    """
    txt = []
    for f in df_files['path']:
        txt += [remote_prefix + f, f]  # Format each file pair for Aspera
    txt = '\n'.join(txt)  # Join file pairs into a single string

    with open(filename, 'w') as fh:
        fh.write(txt)  # Write the string to the file

def format_ascp_command(ascp, pairlist, local='.'):
    """Generate the Aspera download command.

    Args:
        ascp (str): Path to the Aspera ascp executable.
        pairlist (str): Path to the file containing the list of files to download.
        local (str): Local directory where files will be saved.

    Returns:
        str: The complete Aspera command to execute.
    """
    ascp_opts = f'-T -l200m -P 33001 -i {aspera_openssh}'  # Aspera options
    pair_opts = f'--file-pair-list={pairlist} --mode=recv --user=bsaspera --host=fasp-beta.ebi.ac.uk'
    return f'{ascp} {ascp_opts} {pair_opts} {local}'  # Construct the command string

def download_from_bioimage_archive(directory, plate='all', dataset='sequencing', 
        well='all', site='all', query=None, ascp=ascp_guess):
    """Download data from the BioImage Archive based on specified filters.

    Args:
        directory (str): Directory to save the downloaded files.
        plate (str): Filter by plate; 'all' means no filter.
        dataset (str): Dataset to filter by.
        well (str): Filter by well; 'all' means no filter.
        site (str): Filter by site; 'all' means no filter.
        query (str): Additional query filter.
        ascp (str): Path to the Aspera ascp executable.
    """
    os.makedirs(directory, exist_ok=True)  # Create the target directory if it doesn't exist

    # Check for Aspera executable
    if not shutil.which(ascp):
        ascp = shutil.which('ascp')
        if ascp is None:
            print(f'Error: Aspera ascp executable not found at {ascp}')
            raise QuitError

    # Build query based on filters
    queries = ['(Dataset == @dataset)']
    if plate != 'all':
        queries.append('(Plate == @plate)')
    if well != 'all':
        queries.append('(Well == @well)')
    if site != 'all': 
        queries.append('(Site == @site)')
    if query is not None:
        queries.append(f'({query})')

    # Read metadata and apply query filters
    df_bia = (pd.read_csv(bia_files, low_memory=False)
     .query('&'.join(queries))
    )

    if df_bia.pipe(len) == 0:
        raise ValueError('No valid images specified.')  # Raise error if no files match the query

    pairlist = f'{directory}/ascp_download_list.txt'
    write_pairlist(df_bia, pairlist)  # Write the list of files to download
    command = format_ascp_command(ascp, pairlist, local=directory)  # Generate the Aspera command

    print(f'Downloading {len(df_bia)} files from the BioImage Archive with the command: {command}')
    try:
        subprocess.check_call(command, shell=True)  # Execute the Aspera command
    except subprocess.CalledProcessError as e:
        print(f'Error in downloading files using {ascp}. This can be a user network issue. '
            'Try a secure network with better connectivity.'
            )
        raise QuitError

class QuitError(Exception):
    """Exception class to handle errors without generating a stack trace in command line apps."""
    pass

if __name__ == '__main__':
    commands = {
        'download_from_bioimage_archive': download_from_bioimage_archive,
    }
    try:
        fire.Fire(commands)  # Use fire to create a command line interface
    except QuitError:
        sys.exit(1)  # Exit with error status
