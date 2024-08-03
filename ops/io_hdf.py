"""
HDF5 and ND2 Image Processing Utilities
This module provides a set of set of functions for file I/O with HDF5 data.
It includes utilities for:

1. HDF5 Image Saving: Function to save image data and metadata to HDF5 files.
2. HDF5 Image Reading: Function to read image data from HDF5 files, with optional bounding box selection.
3. ND2 to HDF5 Conversion: Utility to convert ND2 files to HDF5 format, including metadata extraction and optional Z-projection.

"""

from tables import file
from tables import open_file
import numpy as np
import ops.filenames
from nd2reader import ND2Reader
import pandas as pd

def save_hdf_image(filename, image, pixel_size_um=1, image_metadata=None,
                   array_name='image', chunkshape=-1):
    """
    Save an image to an HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file.
        image (numpy.ndarray): Image data to be saved.
        pixel_size_um (float, optional): Pixel size in micrometers. Default is 1.
        image_metadata (dict or None, optional): Metadata associated with the image. Default is None.
        array_name (str, optional): Name of the array in the HDF5 file. Default is 'image'.
        chunkshape (int, optional): Chunk shape for creating a compressed array. Default is -1 (no compression).

    Returns:
        None
    """
    # Open the HDF5 file in write mode
    hdf_file = open_file(filename, mode='w')
    try:
        if chunkshape == -1:
            # Create an uncompressed array
            hdf_file.create_array('/', array_name, image)
        else:
            # Create a compressed array with chunking
            hdf_file.create_carray('/', array_name, obj=image, chunkshape=chunkshape)
        
        # Get the image node from the HDF5 file
        image_node = hdf_file.get_node('/', name=array_name)
        
        # Set attributes for pixel size and image metadata
        image_node.attrs.element_size_um = np.array([(pixel_size_um,)] * 3).astype(np.float32)
        image_node.attrs.image_metadata = image_metadata
    except:
        print('Error in saving image array to HDF file')
    finally:
        # Close the HDF5 file
        hdf_file.close()

def read_hdf_image(filename, bbox=None, array_name='image'):
    """
    Reads an image from an HDF5 file with a given bounding box.

    Parameters:
        filename (str): Path to the HDF5 file.
        bbox (tuple or None, optional): Bounding box coordinates (top-left and bottom-right).
                                        Default is None (reads the entire image).
        array_name (str, optional): Name of the array in the HDF5 file. Default is 'image'.

    Returns:
        numpy.ndarray or None: Image data read from the HDF5 file.
    """
    # Open the HDF5 file in read mode
    hdf_file = open_file(filename, mode='r')
    try:
        # Get the image node from the HDF5 file
        image_node = hdf_file.get_node('/', name=array_name)
        if bbox is not None:
            # Check if bbox is within image bounds
            i0, j0 = max(bbox[0], 0), max(bbox[1], 0)
            i1, j1 = min(bbox[2], image_node.shape[-2]), min(bbox[3], image_node.shape[-1])
            # Read the image data within the specified bounding box
            image = image_node[..., i0:i1, j0:j1]
        else:
            # Read the entire image data
            image = image_node[...]
    except:
        print('Error in reading image array from HDF file')
        image = None
    finally:
        # Close the HDF5 file
        hdf_file.close()
    return image


def nd2_to_hdf(file, mag='20X', zproject=True, fov_axes='czxy'):
    """
    Converts ND2 files to HDF5 format.

    Parameters:
        file (str): Path to the ND2 file.
        mag (str, optional): Magnification information. Default is '20X'.
        zproject (bool, optional): Whether to perform Z-projection. Default is True.
        fov_axes (str, optional): Axes order for fields of view. Default is 'czxy'.

    Returns:
        None
    """
    # ND2 file pattern for parsing metadata
    nd2_file_pattern = [
        (r'(?P<dataset>.*)/'
         'Well(?P<well>[A-H][0-9]*)_'
         'Channel((?P<channel_1>[^_,]+)(_[^,]*)?)?,?'
         '((?P<channel_2>[^_,]+)(_[^,]*)?)?,?'
         '((?P<channel_3>[^_,]+)(_[^,]*)?)?,?'
         '((?P<channel_4>[^_,]+)(_[^,]*)?)?,?'
         '((?P<channel_5>[^_,]+)(_[^_]*)?)?'
         '_Seq([0-9]+).nd2')
    ]

    # Parse filename to extract metadata
    description = ops.filenames.parse_filename(file, custom_patterns=nd2_file_pattern)
    description['ext'] = 'hdf'
    description['mag'] = mag
    description['subdir'] = 'preprocess'
    description['dataset'] = None

    channels = [ch for key, ch in description.items() if key.startswith('channel')]

    if len(channels) == 1:
        # Remove the first axis if there's only one channel
        fov_axes = fov_axes[1:]

    # Open the ND2 file and read images
    with ND2Reader(file) as images:
        images.iter_axes = 'v'
        images.bundle_axes = fov_axes

        well_metadata = []

        # Iterate over fields of view
        for site, image in zip(images.metadata['fields_of_view'], images):
            if zproject:
                # Perform Z-projection if enabled
                z_axis = fov_axes.find('z')
                image = image.max(axis=z_axis)
            # Generate filename for saving HDF5 image
            filename = ops.filenames.name_file(description, site=str(site))
            # Save HDF5 image
            save_hdf_image(filename, image[:])

            # Collect metadata for each well
            well_metadata.append({
                'filename': ops.filenames.name_file(description, site=str(site)),
                'field_of_view': site,
                'x': images.metadata['x_data'][site],
                'y': images.metadata['y_data'][site],
                'z': images.metadata['z_data'][site],
                'pfs_offset': images.metadata['pfs_offset'][0],
                'pixel_size': images.metadata['pixel_microns']
            })
        
        # Save metadata to a pickle file
        metadata_filename = ops.filenames.name_file(description, tag='metadata', ext='pkl')
        pd.DataFrame(well_metadata).to_pickle(metadata_filename)
