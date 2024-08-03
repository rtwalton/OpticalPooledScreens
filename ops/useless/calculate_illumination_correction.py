from glob import glob
from ops.process import calculate_illumination_correction
from ops.filenames import parse_filename as parse, name_file as name
from ops.io import save_stack as save
import fire

def well_icf(files, custom_pattern=None, threading=-3, tqdm=True, **kwargs):
    """
    Calculate and save illumination correction for a set of files.

    Parameters:
    -----------
    files : str or list
        Path to the files or a list of file paths for which to calculate the illumination correction.
    custom_pattern : str, optional
        Custom pattern to parse filenames.
    threading : int, default -3
        Number of threads to use for parallel processing.
    tqdm : bool, default True
        Whether to display a progress bar.
    kwargs : dict
        Additional arguments passed to the calculate_illumination_correction function.
    """
    
    # If files is not a list, use glob to find matching files
    if not isinstance(files, list):
        files = glob(files)
    
    # Calculate illumination correction
    icf = calculate_illumination_correction(files, threading=threading, **kwargs)
    
    # Set custom pattern if provided
    if custom_pattern is not None:
        custom_pattern = [custom_pattern]
    
    # Parse description from the first file
    description = parse(files[0], custom_patterns=custom_pattern)
    
    # Save the illumination correction
    save(name(description, subdir='illumination_correction', tag='illumination_correction', site=None), icf)
