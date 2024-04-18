import os
import sys
import shutil
from functools import partial
from multiprocessing import Pool
from glob import iglob
import ops.filenames
import re
import numpy as np
import pandas as pd
import tifffile
from nd2reader import ND2Reader
from joblib import Parallel, delayed


def extract_and_save_metadata(file_list, parse_function_home=None, parse_function_dataset=None):
    """
    Extracts metadata from ND2 files, processes it, and saves it to a pickle file.

    Args:
        file_list (list/str): A list of ND2 file paths or a single ND2 file path.
        parse_function_home (str): Absolute path to the screen directory.
        parse_function_dataset (str): Dataset name within the screen directory.

    Returns:
        None
    """

    # Ensure file_list is a list
    if not isinstance(file_list, list):
        file_list = [file_list]

    # Iterate over each file in the provided list
    for file in file_list:
        # Add parse_filename function to get info from nd2 name and convert to tif filename
        info = parse_file(file, home = parse_function_home, dataset = parse_function_dataset)

        file_description = {}
        # Sort and save info in file_description dictionary
        for k, v in sorted(info.items()):
            file_description[k] = v
        # Add 'subdir' key to file_description
        file_description['subdir'] = 'metadata/'

        # Open ND2 file using ND2Reader
        with ND2Reader(file) as images:
            images.iter_axes = 'v'
            axes = 'xy'
            if 'c' in images.axes:
                axes = 'c' + axes
            if 'z' in images.axes:
                axes = 'z' + axes
            images.bundle_axes = axes


            # Extract raw metadata
            raw_metadata = images.parser._raw_metadata

            # Prepare data dictionary for DataFrame
            data = {
                'x_data': raw_metadata.x_data,
                'y_data': raw_metadata.y_data,
                'z_data': images.metadata['z_coordinates'],
                'pfs_offset': raw_metadata.pfs_offset,
            }

            # Create DataFrame
            df = pd.DataFrame(data)

            # If z_levels metadata exists, select every 4th row
            if 'z_levels' in images.metadata and set(images.metadata['z_levels']) == set(range(0, 4)):
                df = df.iloc[::4, :]

            # Add field_of_view and filename columns to DataFrame
            df['field_of_view'] = images.metadata['fields_of_view']
            df['filename'] = df.apply(lambda row: ops.filenames.name_file(file_description, site=str(row['field_of_view'])), axis=1)

            # Check if the length of raw_metadata matches expected length
            expected_length = len(images.metadata['fields_of_view'])
            if df.shape[0] != expected_length:
                print(f"Warning: Expected {expected_length} fields of view, but got {df.shape[0]}. Skipping metadata extraction.")
                continue

            # Save DataFrame to pickle file
            metadata_filename = ops.filenames.name_file(file_description, tag='metadata', ext='pkl')
            pd.DataFrame(df).to_pickle(metadata_filename)

            
def convert_to_tif(file_list, wells=None, tiles=None, parse_function_home=None, parse_function_dataset=None, zstacks=1):
    """
    Converts a list of image files to TIFF format, considering specified conditions like 'well,' 'tile,' and 'zstacks.'
    This function uses the ND2Reader library to process the files and can handle multiple fields of view (FoVs) or frames
    within the same file, saving them as separate TIFF files with ImageJ metadata.

    Args:
        file_list (list): A list of image file paths.
        wells (list): List of wells to include. Defaults to None, which includes all wells.
        tiles (list): List of tiles to include. Defaults to None, which includes all tiles.
        parse_function_home (str): Absolute path to the screen directory.
        parse_function_dataset (str): Dataset name within the screen directory.
        zstacks (int): Number of z-stacks for the output. Defaults to 1.

    Returns:
        None
    """
    # Check if the input is a list; if not, make it a list with one element.
    if not isinstance(file_list, list):
        file_list = [file_list]

    # Iterate over each file in the provided list.
    for file in file_list:
        # Change the current working directory to the directory containing the current file.
        os.chdir(os.path.dirname(file))

        # Extract file description using the provided parse_function.
        file_description = parse_file(file, home = parse_function_home, dataset = parse_function_dataset)


        # Open the file using ND2Reader.
        with ND2Reader(file) as images:
            # Set the bundle axes for image processing.
            axes = 'xy'
            available_axes = images.iter_axes

            # Check if there are multiple channels in the images.
            if len(images.metadata['channels']) > 1:
                axes = 'c' + axes
                images.bundle_axes = axes
                images.iter_axes = 'v'

                # Iterate over each image and process it.
                for image, v in zip(images, images.metadata['fields_of_view']):
                    # Check if the tile information matches the specified tiles.
                    if tiles is None or v in tiles:
                        file_description['tile'] = v
                    # Generate the output filename for Dapi channel.
                    output_filename = ops.filenames.name_file_channel(file_description, site=str(v), channel="['Dapi_1p']")
                    # Save the image as a TIFF file with ImageJ metadata.
                    tifffile.imwrite(output_filename, np.array(image[0], dtype=np.uint16), metadata={'axes': 'YX'}, imagej=True)
                    print(output_filename)

                # Iterate over each image again to process other channels.
                for image, v in zip(images, images.metadata['fields_of_view']):
                    if tiles is None or v in tiles:
                        file_description['tile'] = v
                    file_description['subdir'] = 'input_sbs/process/input/10X/multidimensional'
                    output_filename = ops.filenames.name_file_channel(file_description, site=str(v))
                    print("Shape:", np.array(image, dtype=np.uint16).shape)
                    tifffile.imwrite(output_filename, np.array(image, dtype=np.uint16), metadata={'axes': 'CYX'} , imagej=True)
                    print(output_filename)
            else:
                # Check if 'z' dimension exists in the images.
                if 'z' in images.axes:
                    axes = 'z' + axes
                    images.bundle_axes = axes
                    images.iter_axes = 'v'

                    # Iterate over each image and process it.
                    for image, v in zip(images, images.metadata['fields_of_view']):
                        if tiles is None or v in tiles:
                            file_description['tile'] = v
                        image = image.max(axis=0)
                        file_description['channel'] = images.metadata['channels']
                        output_filename = ops.filenames.name_file_channel(file_description, site=str(v), channel=images.metadata['channels'])
                        tifffile.imwrite(output_filename, np.array(image, dtype=np.uint16), metadata={'axes': 'YX'}, imagej=True)

                        print(output_filename)
                else:
                    # Iterate over each image and process it.
                    for image, v in zip(images, images.metadata['fields_of_view']):
                        if tiles is None or v in tiles:
                            file_description['tile'] = v
                        file_description['channel'] = images.metadata['channels']
                        output_filename = ops.filenames.name_file_channel(file_description, site=str(v), channel=images.metadata['channels'])
                        rotated_image = np.transpose(image, (1, 0))
                        tifffile.imwrite(output_filename, np.array(rotated_image, dtype=np.uint16), metadata={'axes': 'YX'}, imagej=True)
                        print(output_filename)

def parallel_convert(file_list, wells=None, tiles=None, parse_function_home=None, parse_function_dataset=None, n_jobs=2):
    """
    Function to parallelize the convert_to_tif function.

    Args:
        file_list (list): A list of image file paths.
        wells (list): List of wells to include. Defaults to None, which includes all wells.
        tiles (list): List of tiles to include. Defaults to None, which includes all tiles.
        parse_function_home (str): Absolute path to the screen directory.
        parse_function_dataset (str): Dataset name within the screen directory.
        n_jobs (int): Number of parallel jobs to run. Defaults to 2.

    Returns:
        None
    """
    # Create a partial function based on convert_to_tif with specified parameters.
    fn = partial(convert_to_tif, wells=wells, tiles=tiles, parse_function_home=parse_function_home, parse_function_dataset=parse_function_dataset)

    # Use the Parallel function to parallelize the execution of fn on each file in file_list.
    # n_jobs specifies the number of parallel jobs to run.
    Parallel(n_jobs=n_jobs)(delayed(fn)(f) for f in file_list)


def convert_to_multidimensional_tiff_ph(file_directory, channel_order=None):
    """
    Converts a collection of phenotype TIFF images to a multidimensional TIFF.

    Args:
        file_directory (str): Directory containing TIFF images.
        channel_order (list): List specifying the order of channels. Defaults to None.

    Returns:
        None
    """

    # A dictionary to group files by their common parts
    file_groups = {}

    # Get the path to the 'multidimensional' subdirectory
    multidimensional_dir = os.path.join(file_directory, 'multidimensional')

    # Check if the 'multidimensional' subdirectory exists, and create it if it doesn't
    if not os.path.exists(multidimensional_dir):
        os.makedirs(multidimensional_dir)

    # Get the existing files in the 'multidimensional' subdirectory
    existing_files = set(os.listdir(multidimensional_dir))

    # Search for TIFF files in the specified directory, excluding those in 'multidimensional'
    for root, _, files in os.walk(file_directory):
        for file_name in files:
            if file_name.lower().endswith('.tif') and file_name not in existing_files:
                file_path = os.path.join(root, file_name)
                # Extract common parts of the filename (excluding 'Channel' and 'ext')
                common_parts = '.'.join(file_name.split('.')[:-2])
                if common_parts not in file_groups:
                    file_groups[common_parts] = []
                file_groups[common_parts].append(file_path)
    
    # Process each group of files with the same name
    for file_name, group_files in file_groups.items():
        images = []  # A list to store individual images in the group
        print(f"Processing group: {file_name}")

        if channel_order:
            # Sort the group files based on the specified channel order
            group_files.sort(key=lambda path: channel_order.index(extract_channel(path)))

        # Load individual images and append them to the images list
        for file_path in group_files:
            try:
                image = tifffile.imread(file_path)
                images.append(image)
            except Exception as e:
                print(f"Error reading file: {file_path}")
                print(f"Error message: {e}")
            # Print the channel being appended
            channel = extract_channel(file_path)
            print(f"Appending channel: {channel}")

        # Create a multidimensional array by stacking images along the last axis (channels)
        multidimensional_array = np.stack(images, axis=-1)
        print("Shape before rearranging:", multidimensional_array.shape)
        multidimensional_array = np.transpose(multidimensional_array, (2, 0, 1))
        print("Shape after rearranging:", multidimensional_array.shape)        

        # Save the multidimensional TIFF
        output_filename = os.path.join(multidimensional_dir, f"{file_name}.tif")
        tifffile.imwrite(output_filename, multidimensional_array, metadata={'axes': 'CYX'}, imagej=True)
        print(f"Saved: {output_filename}")
        

def convert_to_multidimensional_tiff_sbs(file_directory, channel_order=None):
    """
    Converts a collection of SBS TIFF images to a multidimensional TIFF.

    Args:
        file_directory (str): Directory containing SBS TIFF images.
        channel_order (list): List specifying the order of channels. Defaults to None.

    Returns:
        None
    """

    file_groups = {}  # A dictionary to group files by their common parts

    # Get the path to the 'multidimensional' subdirectory
    multidimensional_dir = os.path.join(file_directory, 'multidimensional')

    # Check if the 'multidimensional' subdirectory exists, and create it if it doesn't
    if not os.path.exists(multidimensional_dir):
        os.makedirs(multidimensional_dir)

    existing_files = set(os.listdir(multidimensional_dir))

    # Search for TIFF files in the specified directory, excluding those in 'multidimensional'
    for root, _, files in os.walk(file_directory):
        for file_name in files:
            if file_name.lower().endswith('.tif') and file_name not in existing_files:
                if 'c1' not in file_name.lower():
                    file_path = os.path.join(root, file_name)
                    # Extract common parts of the filename (excluding 'Channel' and 'ext')
                    common_parts = '.'.join(file_name.split('.')[:-2])
                    if common_parts not in file_groups:
                        file_groups[common_parts] = []
                    file_groups[common_parts].append(file_path)

                if 'c1' not in common_parts:
                    dapi_c1_file = None
                    # Use a regular expression to replace 'c2' through 'c20' with 'c1'
                    common_parts_corrected_cycle = re.sub(r'c[2-9]|c1[0-9]', 'c1', common_parts)
                    corrected_common_parts = re.sub(r'SBS-[2-9]|SBS-1[0-9]', 'SBS-1', common_parts_corrected_cycle)
                    dapi_c1_file = os.path.join(root, corrected_common_parts + ".Channel-['Dapi_1p'].tif")
                    if os.path.exists(dapi_c1_file) and dapi_c1_file not in file_groups[common_parts]:
                        file_groups[common_parts].append(dapi_c1_file)

    # Process each group of files
    for file_name, group_files in file_groups.items():
        images = []  # A list to store individual images in the group
        print(f"Processing group: {file_name}")
        print(group_files)

        if channel_order:
            # Sort the group files based on the specified channel order
            group_files.sort(key=lambda path: channel_order.index(extract_channel(path)))

        # Load individual images and append them to the images list
        for file_path in group_files:
            try:
                image = tifffile.imread(file_path)
                images.append(image)
            except Exception as e:
                print(f"Error reading file: {file_path}")
                print(f"Error message: {e}")
            # Print the channel being appended
            channel = extract_channel(file_path)
            print(f"Appending channel: {channel}")

        # Create a multidimensional array by stacking images along the last axis (channels)
        multidimensional_array = np.stack(images, axis=-1)
        print("Shape before rearranging:", multidimensional_array.shape)
        multidimensional_array = np.transpose(multidimensional_array, (2, 0, 1))
        print("Shape after rearranging:", multidimensional_array.shape)

        # Save the multidimensional TIFF
        output_filename = os.path.join(multidimensional_dir, f"{file_name}.tif")
        tifffile.imwrite(output_filename, multidimensional_array, metadata={'axes': 'CYX'}, imagej=True)
        print(f"Saved: {output_filename}")

# Function definition to convert a collection of TIFF images to a multidimensional TIFF for 10x cycles
def convert_to_multidimensional_tiff_cycles(file_directory, channel_order=None):
    file_groups = {}  # A dictionary to group files by their common parts

    # Get the path to the 'multidimensional' subdirectory
    multidimensional_dir = os.path.join(file_directory, 'multidimensional')

    # Check if the 'multidimensional' subdirectory exists, and create it if it doesn't
    if not os.path.exists(multidimensional_dir):
        os.makedirs(multidimensional_dir)

    existing_files = set(os.listdir(multidimensional_dir))
    
    # Search for TIFF files in the specified directory, excluding those in 'multidimensional'
    for root, _, files in os.walk(file_directory):
        for file_name in files:
            if file_name.lower().endswith('.tif') and file_name not in existing_files:
                if 'c1' not in file_name.lower():
                    file_path = os.path.join(root, file_name)
                    # Extract common parts of the filename (excluding 'Channel' and 'ext')
                    common_parts = '.'.join(file_name.split('.')[:-2])
                    if common_parts not in file_groups:
                        file_groups[common_parts] = []
                    file_groups[common_parts].append(file_path)
                # file_path = os.path.join(root, file_name)
                # # Extract common parts of the filename (excluding 'Channel' and 'ext')
                # common_parts = '.'.join(file_name.split('.')[:-2])
                # if common_parts not in file_groups:
                #     file_groups[common_parts] = []
                # file_groups[common_parts].append(file_path)

                if 'c1' not in common_parts:
                    dapi_c1_file = None
                    # Use a regular expression to replace 'c2' through 'c20' with 'c1'
                    common_parts_corrected_cycle = re.sub(r'c[2-9]|c1[0-9]', 'c1', common_parts)
                    corrected_common_parts = re.sub(r'SBS-[2-9]|SBS-1[0-9]', 'SBS-1', common_parts_corrected_cycle)
                    dapi_c1_file = os.path.join(root, corrected_common_parts + ".Channel-['Dapi_1p'].tif")
                    if os.path.exists(dapi_c1_file) and dapi_c1_file not in file_groups[common_parts]:
                        file_groups[common_parts].append(dapi_c1_file)

    # Process each group of files
    for file_name, group_files in file_groups.items():
        images = []  # A list to store individual images in the group
        print(f"Processing group: {file_name}")
        print(group_files)

        if channel_order:
            # Sort the group files based on the specified channel order
            group_files.sort(key=lambda path: channel_order.index(extract_channel(path)))
                    
        # Load individual images and append them to the images list
        for file_path in group_files:
            try:
                image = tifffile.imread(file_path)
                images.append(image)
            except Exception as e:
                print(f"Error reading file: {file_path}")
                print(f"Error message: {e}")
            # Print the channel being appended
            channel = extract_channel(file_path)
            print(f"Appending channel: {channel}")

        # Create a multidimensional array by stacking images along the last axis (channels)
        multidimensional_array = np.stack(images, axis=-1)
        # multidimensional_array = np.transpose(multidimensional_array, (2, 1, 0))
        multidimensional_array = np.transpose(multidimensional_array, (2, 0, 1))

        print(multidimensional_array.shape)

        # Save the multidimensional TIFF
        output_filename = os.path.join(multidimensional_dir, f"{file_name}.tif")
        tifffile.imwrite(output_filename, multidimensional_array,metadata={'axes': 'CYX'},imagej=True)
        print(f"Saved: {output_filename}")


def extract_cycle(filename):
    """
    Extracts the cycle number from a filename assuming that the file
    has a subdirectory /c*/ in its name.

    Args:
        filename (str): The filename containing the cycle information.

    Returns:
        str: The extracted cycle number.
    """
    cycle_loc = filename.find('/c')
    cycle = str(filename[cycle_loc + 2: filename.find('/', cycle_loc + 1)])
    cycle_str = 'c' + cycle + '-SBS-' + cycle
    return cycle_str

def extract_well(full_filename):
    """
    Extracts the Well from the filename assuming that the file has WellXX_ in its name.

    Args:
        full_filename (str): The full filename containing the well information.

    Returns:
        str: The extracted well information.
    """
    short_fname = full_filename.split('/')[-1]
    well_loc = short_fname.find('Well')
    well = str(short_fname[well_loc+4:short_fname.find('_', well_loc)])
    return well

def extract_tile(full_filename):
    """
    For files in which the ND2 is split up by FoV, extracts the tile by searching for Seq#### in the filename.

    Args:
        full_filename (str): The full filename containing the tile information.

    Returns:
        str: The extracted tile information.
    """
    short_fname = full_filename.split('/')[-1]
    seq_loc = short_fname.find('Seq')
    seq = str(short_fname[seq_loc+3:seq_loc+7])
    return seq

def extract_plate(full_filename):
    """
    Extracts the plate information from the filename.

    Args:
        full_filename (str): The full filename containing the plate information.

    Returns:
        str: The extracted plate information.
    """
    plate = full_filename.split('/')[5]
    return plate

def extract_channel(file_name):
    """
    Extracts the channel information from the filename.

    Args:
        file_name (str): The filename containing the channel information.

    Returns:
        str: The extracted channel information.
    """
    match = re.search(r"\['(.*?)'\]", file_name)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def parse_file(filename, home, dataset):
    """
    Extracts relevant information from file

    Args:
        filename (str): The filename to parse.
        home (str): The home directory path.
        dataset (str): The dataset name.

    Returns:
        dict: A dictionary containing parsed information from the filename.
    """
    file_description = {}
    file_description['home'] = home
    file_description['dataset'] = dataset
    file_description['ext'] = 'tif'
    file_description['well'] = extract_well(filename)

    if 'input_ph' in filename.split('/'):
        file_description['mag'] = '20X'
        file_description['tag'] = 'phenotype'
        file_description['subdir'] = f"{extract_plate(filename)}/process/input/{file_description['mag']}"
    elif 'input_sbs' in filename.split('/'):
        file_description['mag'] = '10X'
        file_description['tag'] = 'sbs'
        file_description['cycle'] = extract_cycle(filename)
        file_description['subdir'] = f"{extract_plate(filename)}/process/input/{file_description['mag']}"

    return file_description
