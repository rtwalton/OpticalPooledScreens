from tables import *
from tables import file
import numpy as np
import ops.filenames
from nd2reader import ND2Reader
import pandas as pd

def save_hdf_image(filename,image,pixel_size_um=1,image_metadata=None,array_name='image'):
	hdf_file = open_file(filename,mode='w')
	try:
		hdf_file.create_array('/',array_name,image)
		image_node = hdf_file.get_node('/',name=array_name)
		image_node.attrs.element_size_um = np.array([(pixel_size_um,)]*3).astype(np.float32)
		image_node.attrs.image_metadata = image_metadata
	except:
		print('error in saving image array to hdf file')
	hdf_file.close()

def read_hdf_image(filename,bbox=None,array_name='image'):
	"""reads in image from hdf file with given bbox. significantly (~100x) faster when reading in a
	100x100 pixel chunk compared to reading in an entire 1480x1480 tif.
	"""
	hdf_file = open_file(filename,mode='r')
	try:
		image_node = hdf_file.get_node('/',name=array_name)
		if bbox is not None:
			return image_node[...,bbox[0]:bbox[2],bbox[1]:bbox[3]]
		else:
			return image_node[:]
	except:
		print('error in reading image array from hdf file')
	hdf_file.close()

def nd2_to_hdf(file,mag='20X',zproject=True,fov_axes='czxy'):
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

    description = ops.filenames.parse_filename(file,custom_patterns=nd2_file_pattern)
    description['ext']='hdf'
    description['mag']=mag
    description['subdir']='preprocess'

    channels = [ch for key,ch in description.items() if key.startswith('channel')]

    if len(channels)==1:
        fov_axes='xy'

    with ND2Reader(file) as images:
        images.iter_axes='v'
        images.bundle_axes = fov_axes

        well_metadata = []

        for site,image in zip(images.metadata['fields_of_view'],images):
            if zproject:
                image = image.max(axis=1)
            filename = ops.filenames.name_file(description,site=str(site))
            save_hdf_image(filename,image[:])

        well_metadata = [{
                            'filename':ops.filenames.name_file(description,site=str(site)),
                            'field_of_view':site,
                            'x':images.metadata['x_data'][site],
                            'y':images.metadata['y_data'][site],
                            'z':images.metadata['z_data'][site],
                            'pfs_offset':images.metadata['pfs_offset'][0],
                            'pixel_size':images.metadata['pixel_microns']
                        } for site in images.metadata['fields_of_view']]
        metadata_filename = ops.filenames.name_file(description,tag='metadata',ext='pkl')
        pd.DataFrame(well_metadata).to_pickle(metadata_filename)
