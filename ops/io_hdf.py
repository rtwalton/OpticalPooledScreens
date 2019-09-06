from tables import *
from tables import file
import numpy

def save_hdf_image(filename,image,image_metadata=None,array_name='image'):
	# try:
	hdf_file = open_file(filename,mode='a')
	# except:
		# return file._FileRegistry().filenames
	hdf_file.create_array('/',array_name,image)
	# hdf_file[array_name].attrs.image_metadata = image_metadata
	hdf_file.close()