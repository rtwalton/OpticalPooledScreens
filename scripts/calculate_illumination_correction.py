from glob import glob
from ops.process import calculate_illumination_correction
from ops.filenames import parse_filename as parse, name_file as name
from ops.io import save_stack as save
# from joblib import Parallel,delayed
import fire

# TODO: restrict to same acquisition/channels

def well_icf(files, custom_pattern=None, threading=-3, tqdm=True, **kwargs):
	if not isinstance(files,list):
		files = glob(files)
	icf = calculate_illumination_correction(files,threading=threading,tqdm=tqdm,**kwargs)
	if custom_pattern is not None:
		custom_pattern=[custom_pattern]
	description = parse(files[0],custom_patterns=custom_pattern)
	save(name(description,subdir='illumination_correction',tag='illumination_correction',site=None),icf)

# Parallel(n_jobs=-1)(delayed(well_icf)(files) for files in c1_files)


if __name__ == '__main__':
	fire.Fire(well_icf)