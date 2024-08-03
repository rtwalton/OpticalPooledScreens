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

def extract_and_save_metadata(file_list, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=False):
    """
    Extracts metadata from ND2 files, processes it, and saves it to a pickle file.

    Args:
        file_list (list/str): A list of ND2 file paths or a single ND2 file path.
        parse_function_home (str): Absolute path to the screen directory.
        parse_function_dataset (str): Dataset name within the screen directory.
        parse_function_tiles (bool): Whether to include tile information in the parsing function.

    Returns:
        None
    """

    # Ensure file_list is a list
    if not isinstance(file_list, list):
        file_list = [file_list]

    # List to accumulate dataframes for single-tile images
    single_tile_dfs = []

    # Iterate over each file in the provided list
    for file in file_list:
        print(file)
        # Extract info from the file name and convert to tif filename
        info = parse_file(file, home=parse_function_home, dataset=parse_function_dataset, tiles=parse_function_tiles)
        
        # Create and sort file description dictionary
        file_description = {k: v for k, v in sorted(info.items())}
        file_description['subdir'] = 'metadata/'
        
        # Open ND2 file using ND2Reader
        with ND2Reader(file) as images:
            is_multi_tile = isinstance(images.metadata['fields_of_view'], range)
            if is_multi_tile:
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

            # Add additional metadata to DataFrame
            if is_multi_tile:
                df['field_of_view'] = images.metadata['fields_of_view']
                df['filename'] = df.apply(lambda row: ops.filenames.name_file(file_description, site=str(row['field_of_view'])), axis=1)
                # Check if the length of raw_metadata matches expected length
                expected_length = len(images.metadata['fields_of_view'])
                if df.shape[0] != expected_length:
                    print(f"Warning: Expected {expected_length} fields of view, but got {df.shape[0]}. Skipping metadata extraction.")
                    continue
                # Save DataFrame to pickle file
                metadata_filename = ops.filenames.name_file(file_description, tag='metadata', ext='pkl')
                print(metadata_filename)
                df.to_pickle(metadata_filename)
            else:
                df['field_of_view'] = [file_description['tile']] * len(df)
                df['well'] = [file_description['well']] * len(df)
                df['tag'] = [file_description['tag']] * len(df)
                df['cycle'] = [file_description.get('cycle', None)] * len(df)
                df['filename'] = df.apply(lambda row: ops.filenames.name_file(file_description, site=str(row['field_of_view'])), axis=1)
                # print(df)
                single_tile_dfs.append(df)

    # Concatenate and save the single-tile DataFrames
    if single_tile_dfs:
        single_tile_df = pd.concat(single_tile_dfs, ignore_index=True)
        
        # Split by tag: phenotype or sbs
        for tag_value in ['phenotype', 'sbs']:
            subset_df = single_tile_df[single_tile_df['tag'] == tag_value]
            
            if tag_value == 'phenotype':
                # Drop tag and cycle columns
                subset_df = subset_df.drop(columns=['tag', 'cycle'])

                # Group by well and save each group
                grouped = subset_df.groupby('well')
                
                for well, group in grouped:
                    # Sort by field_of_view
                    group['field_of_view'] = pd.to_numeric(group['field_of_view'])
                    group = group.sort_values(by='field_of_view')

                    print(group)                    
                    # Create a unique filename for each group
                    group_file_description = file_description.copy()
                    group_file_description.update({'well': well, 'tile': None})
                    metadata_filename = ops.filenames.name_file(group_file_description, tag='metadata', ext='pkl')
                    
                    print(metadata_filename)
                    # Uncomment the following line to actually save the DataFrame
                    group.to_pickle(metadata_filename)
            
            elif tag_value == 'sbs':
                # Group by cycle and well, then save each group
                grouped = subset_df.groupby(['cycle', 'well'])
                
                for (cycle, well), group in grouped:
                    # Sort by field_of_view
                    group['field_of_view'] = pd.to_numeric(group['field_of_view'])
                    group = group.sort_values(by='field_of_view')
                    print(group)
                    # Create a unique filename for each group
                    group_file_description = file_description.copy()
                    group_file_description.update({'cycle': cycle, 'well': well, 'tile': None})
                    metadata_filename = ops.filenames.name_file(group_file_description, tag='metadata', ext='pkl')
                    
                    print(metadata_filename)
                    group.to_pickle(metadata_filename)


            
def convert_to_tif(file_list, wells=None, tiles=None, channel_order_flip=False, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=False, zstacks=1):
    """
    Converts a list of ND2 image files to TIFF format, accommodating specified criteria such as wells, tiles, and z-stacks.
    This function utilizes the ND2Reader library to process the images, supporting multiple fields of view (FoVs) or frames
    within each file. Each processed image is saved as a separate TIFF file with ImageJ-compatible metadata.

    Args:
        file_list (list): A list of paths to ND2 image files to be converted.
        wells (list): Optional; specifies which wells to include in the conversion. If None, all wells are included.
        tiles (list): Optional; specifies which tiles to include. If None, all tiles are processed.
        channel_order_flip (bool): Optional; if True, reverses the order of channels in the output images. Defaults to False.
        parse_function_home (str): Required; the absolute path to the screen directory for file parsing.
        parse_function_dataset (str): Required; the dataset name within the screen directory for file parsing.
        parse_function_tiles (bool): Optional; specifies whether to include tile parsing. Defaults to False.
        zstacks (int): Optional; specifies the number of z-stacks for output images. Defaults to 1.

    Returns:
        None: The function saves the converted TIFF images to the specified directory structure.
    """
    # Check if the input is a list; if not, make it a list with one element.
    if not isinstance(file_list, list):
        file_list = [file_list]

    # Iterate over each file in the provided list.
    for file in file_list:
        # Change the current working directory to the directory containing the current file.
        os.chdir(os.path.dirname(file))

        # Extract file description using the provided parse_function.
        file_description = parse_file(file, home = parse_function_home, dataset = parse_function_dataset, tiles = parse_function_tiles)
        
        print(file_description)

        # Open the file using ND2Reader.
        with ND2Reader(file) as images:
            # Set the bundle axes for image processing.
            axes = 'xy'
            available_axes = images.iter_axes
            print(images.metadata['channels'])

            # For multichannel images
            if len(images.metadata['channels']) > 1:
                axes = 'c' + axes
                images.bundle_axes = axes
                
                # For multi-tile images
                if isinstance(images.metadata['fields_of_view'], range):
                    images.iter_axes = 'v'

                    # Iterate over each image and process it.
                    for image, v in zip(images, images.metadata['fields_of_view']):
                        # Check if the tile information matches the specified tiles.
                        if tiles is None or v in tiles:
                            file_description['tile'] = v
                        # Generate the output filename for Dapi channel.
                        file_description['subdir'] = 'input_sbs/process/input/10X'
                        output_filename = ops.filenames.name_file_channel(file_description, site=str(v), channel="['Dapi_1p']")
                        # Save the image as a TIFF file with ImageJ metadata.
                        tifffile.imwrite(output_filename, np.array(image[0], dtype=np.uint16), metadata={'axes': 'YX'}, imagej=True)
                        print('Treating as multichannel sbs image, saving Dapi channel', output_filename)
                        # Generate the output filename for multichannel image.
                        file_description['subdir'] = 'input_sbs/process/input/10X/multidimensional'
                        output_filename = ops.filenames.name_file_channel(file_description, site=str(v))
                        print("Shape:", np.array(image, dtype=np.uint16).shape)
                        tifffile.imwrite(output_filename, np.array(image, dtype=np.uint16), metadata={'axes': 'CYX'} , imagej=True)
                        print('Treating as multichannel sbs image, saving all channels', output_filename)
                # For single-tile images                       
                else:
                    image=images[0]
                    output_filename = ops.filenames.name_file_channel(file_description)
                    if channel_order_flip:
                        image = np.flip(image, axis=0)
                    print("Shape:", np.array(image, dtype=np.uint16).shape)
                    tifffile.imwrite(output_filename, np.array(image, dtype=np.uint16), metadata={'axes': 'CYX'} , imagej=True)
                    print('Treating as multichannel image, saving all channels', output_filename)
            
            # For single channel images
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
                        print('Treating as single-channel z-maxxed ph image, saving one channel', output_filename)
                else:
                    # Iterate over each image and process it.
                    for image, v in zip(images, images.metadata['fields_of_view']):
                        if tiles is None or v in tiles:
                            file_description['tile'] = v
                        file_description['channel'] = images.metadata['channels']
                        output_filename = ops.filenames.name_file_channel(file_description, site=str(v), channel=images.metadata['channels'])
                        rotated_image = np.transpose(image, (1, 0))
                        tifffile.imwrite(output_filename, np.array(rotated_image, dtype=np.uint16), metadata={'axes': 'YX'}, imagej=True)
                        print('Treating as single-channel sbs image, saving one channel', output_filename)

def parallel_convert(file_list, wells=None, tiles=None, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=False, n_jobs=2):
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
    fn = partial(convert_to_tif, wells=wells, tiles=tiles, parse_function_home=parse_function_home, parse_function_dataset=parse_function_dataset,parse_function_tiles=parse_function_tiles)

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
    Extracts the Well from the filename assuming that the file has Wells-XX_ or WellXX_ in its name.

    Args:
        full_filename (str): The full filename containing the well information.

    Returns:
        str: The extracted well information.
    """
    short_fname = full_filename.split('/')[-1]
    if 'Wells-' in short_fname:
        well_loc = short_fname.find('Wells-')
        well_prefix_length = 6  # Length of 'Wells-'
    else:
        well_loc = short_fname.find('Well')
        well_prefix_length = 4  # Length of 'Well'
    
    if well_loc == -1:
        raise ValueError("No 'Well' or 'Wells' found in the filename.")
    
    well = str(short_fname[well_loc + well_prefix_length : short_fname.find('_', well_loc)])
    return well


def extract_tile(full_filename):
    """
    For files in which the ND2 is split up by FoV, extracts the tile by searching for Points-#### in the filename.

    Args:
        full_filename (str): The full filename containing the tile information.

    Returns:
        str: The extracted tile information.
    """
    short_fname = full_filename.split('/')[-1]
    seq_loc = short_fname.find('Points-')
    
    if seq_loc == -1:
        raise ValueError("No 'Points-' found in the filename.")
    
    seq_end = short_fname.find('_', seq_loc)
    
    if seq_end == -1:
        raise ValueError("No underscore found after 'Points-' in the filename.")
    
    seq = short_fname[seq_loc + 7: seq_end].lstrip('0')
    
    if seq == '':
        seq = '0'
    
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

def parse_file(filename, home, dataset, tiles=False):
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

    if tiles:
        file_description['tile'] = extract_tile(filename)

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
