from tables import *
from tables import file
import numpy

def save_hdf_image(filename,image,pixel_size_um=1,image_metadata=None,array_name='image'):
	# try:
	hdf_file = open_file(filename,mode='w')
	# except:
		# return file._FileRegistry().filenames
	hdf_file.create_array('/',array_name,image)
	hdf_file[array_name].attrs.element_size_um = (pixel_size_um,)*2
	hdf_file[array_name].attrs.image_metadata = image_metadata
	hdf_file.close()