import re
import os
import time
from urllib.parse import urlparse
from glob import glob
from natsort import natsorted

from ops.constants import FILE

# Regular expressions for parsing file and folder names
FILE_PATTERN = [
        r'((?P<home>.*)\/)?',  # Home directory (optional)
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',  # Dataset and date
        r'(?:(?P<subdir>.*)\/)*',  # Subdirectory (optional)
        r'(MAX_)?(?P<mag>[0-9]+X).',  # Magnification
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?',  # Cycle (optional)
        r'(?P<well>[A-H][0-9]{1,2})',  # Well
        r'([_-](?P<channel_1>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)?',  # Channel 1 (optional)
        r'([_-](?P<channel_2>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)?',  # Channel 2 (optional)
        r'([_-](?P<channel_3>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)?',  # Channel 3 (optional)
        r'([_-](?P<channel_4>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)?',  # Channel 4 (optional)
        r'([_-](?P<channel_5>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)?',  # Channel 5 (optional)
        r'([_-](?P<channel_6>((DAPI-GFP)|(DAPI)|(GFP)|(CY3)|(A594)|(CY5)|(CY7)|(AF750)|(mCherry)))(_(1x1)|([0-9]+p))?)*?',  # Channel 6 (optional)
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',  # Site (optional)
        r'(?:_Tile-(?P<tile>([0-9]+)))?',  # Tile (optional)
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)'  # Tag and extension
]

FOLDER_PATTERN = [
        r'(?P<mag>[0-9]+X).',  # Magnification
        r'(?:(?P<cycle>[^_\.]*).*?)\/',  # Cycle (optional)
        r'(?P<well>[A-H][0-9]+)',  # Well
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',  # Site (optional)
        r'\/?'  # Optional trailing slash
]

# Full file and folder patterns
FILE_PATTERN_ABS = ''.join(FILE_PATTERN)
FILE_PATTERN_REL = ''.join(FILE_PATTERN[2:])

FOLDER_PATTERN_ABS = ''.join(FILE_PATTERN[:2] + FOLDER_PATTERN)
FOLDER_PATTERN_REL = ''.join(FOLDER_PATTERN)


def parse_filename(filename, custom_patterns=None):
    """Parse filename into dictionary.
    
    Args:
        filename (str): The filename to parse.
        custom_patterns (list): Custom regular expression patterns to consider during parsing. Defaults to None.

    Returns:
        dict: A dictionary containing parsed components of the filename.

    Raises:
        ValueError: If the filename cannot be parsed.

    Examples:
        >>> parse_filename('example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif')
        {'subdir': 'example_data/input/10X_c1-SBS-1',
         'mag': '10X',
         'cycle': 'c1-SBS-1',
         'well': 'A1',
         'tile': '107',
         'tag': 'max',
         'ext': 'tif',
         'file': 'example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif'}
    """

    if isinstance(filename,list):
        # If filename is a list, recursively parse each element in the list
        return [parse_filename(file) for file in filename]

    # Normalize path separators and convert to forward slashes for consistency
    filename = normpath(filename)
    # Normalize path separators for Windows
    filename = filename.replace('\\', '/')

    # Define patterns to match against the filename
    patterns = [FILE_PATTERN_ABS, FILE_PATTERN_REL, 
                FOLDER_PATTERN_ABS, FOLDER_PATTERN_REL]

    # Add custom patterns if provided
    if custom_patterns is not None:
        patterns += list(custom_patterns)

    # Iterate through each pattern and attempt to match against the filename
    for pattern in patterns:
        match = re.match(pattern, filename)
        try:
            # Extract matched groups and create a dictionary
            result = {k:v for k,v in match.groupdict().items() if v is not None}
            # Add the original filename for convenience (not used by name_file)
            result[FILE] = filename  
            return result
        except AttributeError:
            # If no match is found, continue to the next pattern
            continue
    
    # Raise an error if no pattern matches the filename
    raise ValueError('Failed to parse filename: %s' % filename)


def name_file(description, **more_description):
    """Name a file based on a dictionary of filename parts with optional overrides.

    Args:
        description (dict): A dictionary containing filename parts.
        **more_description: Additional keyword arguments to override values in the 'description' dictionary.

    Returns:
        str: The generated filename.

    Notes:
        The function constructs a filename using the provided dictionary of filename parts. It supports overriding
        dictionary values with keyword arguments. Some parts of the filename are optional and depend on the presence
        of certain keys in the dictionary.

    Examples:
        >>> description = {'mag': '10X', 'cycle': 'c1-SBS-1', 'well': 'A1', 'tile': '107', 'tag': 'max', 'ext': 'tif'}
        >>> name_file(description)
        '10X_c1-SBS-1_A1_Tile-107.max.tif'

        >>> name_file(description, tag='min')
        '10X_c1-SBS-1_A1_Tile-107.min.tif'
    """
    d = dict(description)
    d.update(more_description)
    # If value is None, key is removed
    d = {k: v for k,v in d.items() if v is not None}

    # Construct the first part of the filename based on magnification and cycle (if present)
    if 'cycle' in d:
        d['first'] = '{mag}_{cycle}_{well}'.format(**d)
    else:
        d['first'] = '{mag}_{well}'.format(**d)

    # Construct the middle part of the filename based on channel information
    channels = [ch for key,ch in natsorted(d.items()) if key.startswith('channel')]
    if len(channels) > 0:
        d['middle'] = '-'.join(channels)

    # Determine the positional information (tile or site) and append to the filename
    if 'tile' in d:
        d['pos'] = 'Tile-{0}'.format(d['tile'])
    elif 'site' in d:
        d['pos'] = 'Site-{0}'.format(d['site'])

    # Define filename formats and attempt to construct the filename
    formats = [
        '{first}_{middle}_{pos}.{tag}.{ext}',
        '{first}_{pos}.{tag}.{ext}',
        '{first}_{middle}_{pos}.{ext}',
        '{first}_{pos}.{ext}',
        '{first}_{middle}.{tag}.{ext}',
        '{first}.{tag}.{ext}',
        '{first}_{middle}.{ext}',
        '{first}.{ext}',
    ]
    for fmt in formats:
        try:
            basename = fmt.format(**d)
            break
        except KeyError:
            continue
    else:
        raise ValueError('Extension missing')

    # Construct the full path of the filename
    optional = lambda x: d.get(x, '')
    filename = os.path.join(optional('home'), optional('dataset'), optional('subdir'), basename)
    return normpath(filename)


def name_file_channel(description, **more_description):
    """Name a file based on a dictionary of filename parts with optional overrides.

    Args:
        description (dict): A dictionary containing filename parts.
        **more_description: Additional keyword arguments to override values in the 'description' dictionary.

    Returns:
        str: The generated filename.

    Notes:
        The function constructs a filename using the provided dictionary of filename parts. It supports overriding
        dictionary values with keyword arguments. Some parts of the filename are optional and depend on the presence
        of certain keys in the dictionary. This version specifically includes a channel section in the filename.

    Examples:
        >>> description = {'mag': '10X', 'cycle': 'c1-SBS-1', 'well': 'A1', 'tile': '107', 'channel': 'DAPI', 'tag': 'max', 'ext': 'tif'}
        >>> name_file_channel(description)
        '10X_c1-SBS-1_A1_Tile-107.max.Channel-DAPI.tif'

        >>> name_file_channel(description, tag='min')
        '10X_c1-SBS-1_A1_Tile-107.min.Channel-DAPI.tif'
    """
    d = dict(description)
    d.update(more_description)
    # If value is None, key is removed
    d = {k: v for k,v in d.items() if v is not None}

    # Construct the first part of the filename based on magnification and cycle (if present)
    if 'cycle' in d:
        d['first'] = '{mag}_{cycle}_{well}'.format(**d)
    else:
        d['first'] = '{mag}_{well}'.format(**d)

    # Determine the positional information (tile or site) and append to the filename
    if 'tile' in d:
        d['pos'] = 'Tile-{0}'.format(d['tile'])
    elif 'site' in d:
        d['pos'] = 'Site-{0}'.format(d['site'])
    else:
        d['pos'] = None
    
    # Check if the 'channel' key exists in the dictionary and format the channel section
    if 'channel' in d:
        d['channel'] = 'Channel-{0}'.format(d['channel'])

    # Define filename formats and attempt to construct the filename
    formats = [
        '{first}_{pos}.{tag}.{channel}.{ext}',  # Include channel here
        '{first}_{pos}.{tag}.{ext}',
        '{first}_{pos}.{ext}',
        '{first}.{tag}.{ext}',
        '{first}.{ext}',
    ]
    for fmt in formats:
        try:
            basename = fmt.format(**d)
            break
        except KeyError:
            continue
    else:
        raise ValueError('Extension missing')

    # Construct the full path of the filename
    optional = lambda x: d.get(x, '')
    filename = os.path.join(optional('home'), optional('dataset'), optional('subdir'), basename)
    return normpath(filename)

def normpath(filename):
    """Normalize a file path.

    Args:
        filename (str): The file path to normalize.

    Returns:
        str: The normalized file path.

    """
    # Check if the filename contains a URL scheme
    if not urlparse(filename).scheme:  # Leave remote URLs alone
        # Normalize the file path using os.path.normpath
        filename = os.path.normpath(filename)
    return filename


def guess_filename(row, tag, **override_fields):
    """Guess a filename based on row data.

    Args:
        row (pd.Series): The row containing data for filename construction.
        tag (str): The tag to be included in the filename.
        **override_fields: Additional keyword arguments to override values in the row data.

    """
    # Default description with 'subdir', 'mag', 'tag', and 'ext' fields
    description = {'subdir': 'process', 'mag': '10X', 'tag': tag, 'ext': 'tif'}
    # Update description with row data and overrides
    description.update(row.to_dict())
    description.update(override_fields)
    # Generate filename based on the description
    return name_file(description)


def make_filename(df, base_description, **kwargs):
    """Make filenames based on DataFrame and base description.

    Args:
        df (pd.DataFrame): The DataFrame containing data for filename construction.
        base_description (dict): The base description for filename construction.
        **kwargs: Additional keyword arguments to override values in the DataFrame and base description.

    Returns:
        list: A list of generated filenames.

    """
    # Make a copy of the base description to avoid modifying the original
    d = base_description.copy()
    arr = []  # Initialize an empty list to store generated filenames
    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Update the base description with data from the current row
        d.update(row.to_dict())
        # Update the base description with any additional overrides provided as keyword arguments
        d.update(kwargs)
        # Generate a filename based on the updated description and append it to the list
        arr.append(name_file(d))
    return arr  # Return the list of generated filenames


def make_filename_pipe(df, output_col, template_or_description=None, **kwargs):
    """Make filenames based on DataFrame and template or description.

    Args:
        df (pd.DataFrame): The DataFrame containing data for filename construction.
        output_col (str): The name of the column to store the generated filenames.
        template_or_description (str or dict): The template or description for filename construction.
        **kwargs: Additional keyword arguments to override values in the DataFrame and description.

    Returns:
        pd.DataFrame: The DataFrame with the generated filenames added as a new column.

    """
    try:
        # Attempt to parse the template or description into a dictionary
        description = parse_filename(template_or_description)
    except TypeError:
        # If the input is not a string or dictionary, assume it's already a description dictionary
        description = template_or_description.copy()

    arr = []  # Initialize an empty list to store generated filenames
    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Update the description with data from the current row
        description.update(row.to_dict())
        # Update the description with any additional overrides provided as keyword arguments
        description.update(kwargs)
        # Generate a filename based on the updated description and append it to the list
        arr.append(name_file(description))

    # Add a new column to the DataFrame containing the generated filenames
    return df.assign(**{output_col: arr})


def timestamp(filename='', fmt='%Y%m%d_%H%M%S', sep='.'):
    """Append a timestamp to a filename.

    Args:
        filename (str): The filename to append the timestamp to.
        fmt (str): The format of the timestamp. Defaults to '%Y%m%d_%H%M%S'.
        sep (str): The separator between the filename and the timestamp. Defaults to '.'.

    Returns:
        str: The filename with the appended timestamp.

    """
    # Generate the timestamp string using the specified format
    stamp = time.strftime(fmt)
    # Define a regular expression pattern to split the filename into two parts: filename and extension
    pat = r'(.*)\.(.*)'
    # Use regex to split the filename into its components
    match = re.findall(pat, filename)
    # If the filename has a valid format and extension
    if match:
        # Construct the new filename with the timestamp inserted between the filename and extension
        return sep.join([match[0][0], stamp, match[0][1]])
    # If the filename is empty or doesn't have a valid format
    elif filename:
        # Append the timestamp to the end of the filename, separated by the specified separator
        return sep.join([filename, stamp])
    else:
        # If the filename is empty, return just the timestamp
        return stamp


def file_frame(files_or_search, **kwargs):
    """Create a DataFrame of filenames from a list of files or a wildcard search term.

    Args:
        files_or_search (str or list): Either a list of files or a glob wildcard search term.
        **kwargs: Additional keyword arguments to pass to `parse_filename`.

    Returns:
        pd.DataFrame: A DataFrame containing parsed filename parts.

    """
    from natsort import natsorted
    import pandas as pd

    # If files_or_search is a string, treat it as a glob wildcard search term
    if isinstance(files_or_search, str):
        # Get a sorted list of files matching the search term
        files = natsorted(glob(files_or_search))
    else:
        # If files_or_search is a list, use it as the list of files
        files = files_or_search

    # Create a DataFrame by applying parse_filename function to each file in the list
    return pd.DataFrame([parse_filename(f, **kwargs) for f in files])
