"""
Plate and Well Coordinate Utilities
This module provides functions for handling and manipulating plate and well coordinates
(relating to step 3 -- merge). It includes functions for:

1. Coordinate Conversion: Converting between different coordinate systems (e.g., well to pixel coordinates).
2. Global Coordinate Calculation: Adding global coordinates to multi-well plate data.
3. Well Identifier Processing: Converting between different well naming conventions and formats.
4. Grid Mapping: Handling different grid layouts and spacings for multi-well plates.
5. Position List Filtering: Processing and filtering Micro-Manager position lists for specific wells and sites.

"""

import string
import re
import numpy as np
import pandas as pd

def to_pixel_xy(df):
    """
    Convert x and y coordinates in a DataFrame to pixel coordinates.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing x and y coordinates and pixel size.

    Returns:
    - pandas.DataFrame: DataFrame with additional columns for x and y coordinates in pixels.
    """
    # Get the reference x and y coordinates (x_0, y_0) from the first row of the DataFrame
    x_0, y_0 = (df.iloc[0].x, df.iloc[0].y)
    
    # Calculate x_px and y_px columns by subtracting the reference coordinates (x_0, y_0)
    # from each x and y coordinate, respectively, and dividing by the pixel size
    df['x_px'] = df.apply(lambda row: (row.x - x_0) / row.pixel_size, axis=1)
    df['y_px'] = df.apply(lambda row: (row.y - y_0) / row.pixel_size, axis=1)

    return df

def add_global_xy(df, well_spacing, grid_shape, grid_spacing='10X', factor=1., snake_remap=False,
    ij=('i', 'j'), xy=('x', 'y'), tile='tile'):
    """
    Adds global x and y coordinates to a DataFrame with columns indicating (i, j) or (x, y) positions.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing positions to be converted to global coordinates.
    - well_spacing (float): Spacing between wells.
    - grid_shape (tuple): Shape of the grid (e.g., (8, 12) for an 8x12 grid).
    - grid_spacing (str, optional): Spacing of the grid. Default is '10X'.
    - factor (float, optional): Scaling factor for converting positions. Default is 1.
    - snake_remap (bool, optional): Whether to use snake remapping. Default is False.
    - ij (tuple, optional): Names of columns indicating (i, j) positions. Default is ('i', 'j').
    - xy (tuple, optional): Names of columns indicating (x, y) positions. Default is ('x', 'y').
    - tile (str, optional): Name of the column indicating tile. Default is 'tile'.

    Returns:
    - pandas.DataFrame: DataFrame with additional columns for global x and y coordinates.
    """
    I, J = ij  # Get column names for (i, j) positions
    X, Y = xy  # Get column names for (x, y) positions
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    wt = list(zip(df['well'], df['tile']))  # Zip well and tile columns together
    # Create a dictionary to map (well, tile) pairs to plate coordinates using plate_coordinate function
    d = {(w, t): plate_coordinate(w, t, well_spacing, grid_spacing, grid_shape, snake_remap) for w, t in set(wt)}
    # Unzip dictionary values to get lists of x and y coordinates
    y, x = zip(*[d[k] for k in wt])

    # Check if DataFrame contains (x, y) or (i, j) positions and add global x and y columns accordingly
    if 'x' in df:
        df['global_x'] = x + df[X] * factor  # Add global x column based on (x, y) positions
        df['global_y'] = y + df[Y] * factor  # Add global y column based on (x, y) positions
    elif 'i' in df:
        df['global_x'] = x + df[J] * factor  # Add global x column based on (i, j) positions
        df['global_y'] = y + df[I] * factor  # Add global y column based on (i, j) positions
    else:
        df['global_x'] = x  # Add global x column if neither (x, y) nor (i, j) positions are found
        df['global_y'] = y  # Add global y column if neither (x, y) nor (i, j) positions are found

    df['global_y'] *= -1  # Invert global y coordinates (assuming y increases downwards)

    return df  # Return the DataFrame with added global x and y coordinates



def plate_coordinate(well, tile, well_spacing, grid_spacing, grid_shape, snake_remap=False):
    """
    Returns global plate coordinate (i, j) in microns for a tile in a well.
    The position is based on:
    - `well_spacing` microns, or one of '96w', '24w', '6w' for standard well plates
    - `grid_spacing` microns, or one of '10X', '20X' for common spacing at a given magnification
    - `grid_shape` (# rows, # columns)

    Parameters:
    - well (str or int): Well identifier.
    - tile (int): Tile number.
    - well_spacing (int or str): Spacing between wells in microns or standard well plate size identifier.
    - grid_spacing (int or str): Spacing between grid elements in microns or common spacing identifier.
    - grid_shape (tuple): Number of rows and columns in the grid.
    - snake_remap (bool, optional): Whether to apply snake remapping. Default is False.

    Returns:
    - tuple: Global plate coordinate (i, j) in microns.
    """
    tile = int(tile)

    # Convert well spacing and grid spacing to integers if they are given as strings
    if isinstance(well_spacing, int):
        well_spacing = well_spacing
    elif well_spacing.upper() == '96W':
        well_spacing = 9000
    elif well_spacing.upper() == '24W':
        well_spacing = 19300
    elif well_spacing.upper() == '6W':
        well_spacing = 39120

    if isinstance(grid_spacing, int):
        delta = grid_spacing
    elif grid_spacing.upper() == '10X':
        delta = 1280
    elif grid_spacing.upper() == '20X':
        delta = 640
    else:
        delta = grid_spacing

    # Apply snake remapping if enabled
    if snake_remap:
        tile = int(remap_snake(tile, grid_shape))

    # Convert well identifier to row and column indices
    row, col = well_to_row_col(well, mit=True)

    # Calculate initial (i, j) position based on row and column indices and well spacing
    i, j = row * well_spacing, col * well_spacing

    # Calculate adjustments based on grid shape and tile number
    height, width = grid_shape
    i += delta * int(tile / width)
    j += delta * (tile % width)

    # Center the coordinates within the grid
    i -= delta * ((height - 1) / 2.)
    j -= delta * ((width  - 1)  / 2.)

    return i, j

def add_row_col(df, well='well', mit=False):
    """
    Adds row and column indices to a dataframe based on the well column.

    Parameters:
    - df (DataFrame): Input dataframe.
    - well (str): Column name indicating the well identifier.
    - mit (bool, optional): Whether to use the MIT-style well identifiers. Default is False.

    Returns:
    - DataFrame: Dataframe with additional 'row' and 'col' columns.
    """
    # Calculate row and column indices using the `well_to_row_col` function for each well identifier in the dataframe
    rows, cols = zip(*[well_to_row_col(w, mit=mit) for w in df[well]])

    # Assign the calculated row and column indices as new columns to the dataframe
    return df.assign(row=rows, col=cols)

def well_to_row_col(well, mit=False):
    """
    Converts a well identifier to row and column indices.

    Parameters:
        well (str): The well identifier.
        mit (bool, optional): Whether to interpret the well identifier in the MIT-style format. 
                              Default is False, which interprets the well identifier in the standard format.

    Returns:
        tuple: A tuple containing the row index and column index.
    """
    if mit:
        # MIT-style well identifier
        # Extract the row index as the index of the first character in the alphabet
        # Extract the column index as the numeric value after the first character, minus 1
        return string.ascii_uppercase.index(well[0]), int(well[1:]) - 1
    else:
        # Standard well identifier
        # Extract the row index as the first character
        # Extract the column index as the numeric value after the first character
        return well[0], int(well[1:])


def standardize_well(df, col='well'):
    """
    Standardizes well labels in a DataFrame to the format 'A01' (row letter followed by a two-digit column number).

    Parameters:
        df (DataFrame): The DataFrame containing the well labels.
        col (str, optional): The name of the column containing the well labels. Default is 'well'.

    Returns:
        DataFrame: The DataFrame with standardized well labels.
    """
    # Extracts the well labels from the specified column and standardizes their format
    arr = ['{0}{1:02d}'.format(w[0], int(w[1:])) for w in df[col]]
    # Assigns the standardized well labels back to the specified column in the DataFrame
    return df.assign(**{col: arr})


def remap_snake(site, grid_shape):
    """
    Maps site names from snake order (Micro-Manager HCS plugin) to row order.

    Parameters:
        site (int or str): The site name in snake order.
        grid_shape (tuple): The shape of the grid in terms of rows and columns.

    Returns:
        str: The site name remapped to row order.
    """
    # Extracting rows and columns from grid_shape tuple
    rows, cols = grid_shape
    # Creating a grid of indices with shape (rows, cols)
    grid = np.arange(rows * cols).reshape(rows, cols)
    # Reversing every second row in the grid
    grid[1::2] = grid[1::2, ::-1]
    # Flattening the grid and retrieving the remapped site index
    site_ = grid.flat[int(site)]
    # Converting the remapped site index to string
    return '%d' % site_


def filter_micromanager_positions(filename, well_site_list):
    """
    Restrict micromanager position list to given wells and sites.

    Parameters:
        filename (str): The filename of the Micromanager position list.
        well_site_list (DataFrame or list of tuples): DataFrame with 'well' and 'site' columns or list of tuples with (well, site) pairs.

    Returns:
        None
    """
    # Convert DataFrame to list of tuples if needed
    if isinstance(well_site_list, pd.DataFrame):
        well_site_list = zip(well_site_list['well'], well_site_list['site'])

    # Convert to set for faster lookup
    well_site_list = set((str(w), str(s)) for w, s in well_site_list)

    # Define a function to filter positions based on well and site
    def filter_well_site(position):
        # Regular expression pattern to extract well and site information from position label
        pat = '(.\d+)-Site_(\d+)'
        # Extracting well and site using regex and checking if it's in the provided list
        return re.findall(pat, position['LABEL'])[0] in well_site_list

    # Read positions from the JSON file
    with open(filename, 'r') as fh:
        d = json.load(fh)
        print('read %d positions from %s' % (len(d['POSITIONS']), filename))
    
    # Filter positions based on well and site
    d['POSITIONS'] = list(filter(filter_well_site, d['POSITIONS']))
    
    # Generate timestamp for the new filename
    timestamp = '{date:%Y%m%d_%I.%M%p}'.format(date=datetime.datetime.now())
    filename2 = '%s.%s.filtered.pos' % (filename, timestamp)
    
    # Write filtered positions to a new JSON file
    with open(filename2, 'w') as fh:
        json.dump(d, fh)
        print('...')
        print('wrote %d positions to %s' % (len(d['POSITIONS']), filename2))
