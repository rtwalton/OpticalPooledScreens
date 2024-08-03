"""
Image Annotation Utilities
This module provides a collection of functions for annotating and processing microscopy images
(relating to SBS base calling -- step 1). It includes functions for:

1. Image Labeling: Creating and manipulating labeled image masks.
2. Text Annotation: Drawing text and labels on images.
3. Base Calling: Annotating and processing DNA base calls in sequencing data.
4. Color Management: Creating and manipulating color lookup tables (LUTs).
5. Geometric Operations: Adding bounds and outlines to image features.

"""

import numpy as np
import pandas as pd
import skimage.morphology
import warnings
from itertools import count
import os
import PIL.Image
import PIL.ImageFont

from ops.constants import *
import ops.filenames
import ops
import ops.io

def load_truetype(truetype='visitor1.ttf', size=10):
    """
    Loads a TrueType font from a specified file path and returns a font object.

    Parameters:
    truetype (str, optional): The name of the TrueType font file to load. Default is 'visitor1.ttf'.
    size (int, optional): The "em" size in pixels, which differs from the actual height of the letters for most fonts. Default is 10.

    Returns:
    PIL.ImageFont.ImageFont: The loaded font object if the font is successfully loaded.

    Raises:
    OSError: If the TrueType font is not found at the specified path, a warning is issued.

    Notes:
    - The `size` parameter defines the "em" size in pixels.
    - The font file is expected to be located in the same directory as the `ops` module.
    """
    PATH = os.path.join(os.path.dirname(ops.__file__), truetype)  # Define the path to the font file
    try:
        return PIL.ImageFont.truetype(PATH, size=size)  # Attempt to load the TrueType font
    except OSError as e:
        warnings.warn('TrueType font not found at {0}'.format(PATH))  # Issue a warning if the font file is not found

# Load the default TrueType font using load_truetype function with default parameters
VISITOR_FONT = load_truetype()

def annotate_labels(df, label, value, label_mask=None, tag='cells', outline=False):
    """
    Transfer `value` from dataframe `df` to a saved integer image mask, using 
    `label` as an index.

    The dataframe should contain data from a single image, which is loaded from
    `label_mask` if provided, or else guessed based on descriptors in the first 
    row of `df` and `tag`.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing label and value data.
    label (str): The column name in `df` to be used as an index for labeling.
    value (str): The column name in `df` whose values will be transferred to the mask.
    label_mask (str or numpy.ndarray, optional): Path to the label mask image file or an array representing the mask. If None, the mask is guessed based on `df` and `tag`.
    tag (str, optional): Tag used to guess the filename of the label mask if `label_mask` is None. Default is 'cells'.
    outline (bool, optional): If True, outline the mask before transferring values. Default is False.

    Returns:
    numpy.ndarray: The labeled phenotype mask with transferred values.

    Raises:
    ValueError: If duplicate rows or non-integer label columns are found, or if duplicate indices are present.

    Notes:
    - The function assumes the DataFrame contains data from a single image.
    - If the value column is not numeric, it is converted to a categorical type with a warning.
    """
    if df[label].duplicated().any():
        raise ValueError('duplicate rows present')  # Check for duplicate rows in the DataFrame

    label_to_value = df.set_index(label, drop=False)[value]  # Map labels to values
    index_dtype = label_to_value.index.dtype
    value_dtype = label_to_value.dtype
    if not np.issubdtype(index_dtype, np.integer):
        raise ValueError('label column {0} is not integer type'.format(label))  # Ensure label column is integer type

    if not np.issubdtype(value_dtype, np.number):
        label_to_value = label_to_value.astype('category').cat.codes  # Convert non-numeric value column to categorical
        warnings.warn('converting value column "{0}" to categorical'.format(value))  # Warn about conversion

    if label_to_value.index.duplicated().any():
        raise ValueError('duplicate index')  # Ensure no duplicate indices

    top_row = df.iloc[0]
    if label_mask is None:
        filename = ops.filenames.guess_filename(top_row, tag)  # Guess the filename based on descriptors in the first row and tag
        labels = ops.io.read_stack(filename)  # Load the label mask
    elif isinstance(label_mask, str):
        labels = ops.io.read_stack(label_mask)  # Load the label mask from the given file path
    else:
        labels = label_mask  # Use the provided label mask array
    
    if outline:
        labels = outline_mask(labels, 'inner')  # Outline the mask if specified
    
    phenotype = relabel_array(labels, label_to_value)  # Transfer values to the mask
    
    return phenotype  # Return the labeled phenotype mask

def annotate_points(df, value, ij=('i', 'j'), width=3, shape=(1024, 1024), selem=None):
    """
    Create a mask with pixels at coordinates `ij` set to `value` from dataframe `df`.
    Dilation is performed with `selem` if provided, or else with a square of `width`.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing coordinates and values.
    value (str): The column name in `df` whose values will be set in the mask.
    ij (tuple, optional): The column names in `df` representing the coordinates. Default is ('i', 'j').
    width (int, optional): The width of the square structuring element if `selem` is not provided. Default is 3.
    shape (tuple, optional): The shape of the output mask. Default is (1480, 1480).
    selem (numpy.ndarray, optional): The structuring element used for dilation. If None, a square of `width` is used.

    Returns:
    numpy.ndarray: The mask with pixels set to values from `df` and dilated.

    Notes:
    - The function converts the coordinates to integers and initializes a mask of the specified shape.
    - If `selem` is not provided, a square structuring element of the given `width` is used for dilation.
    """
        
    ij = df[list(ij)].values.astype(int)  # Extract and convert coordinates to integers
    n = ij.shape[0]  # Get the number of coordinates
    mask = np.zeros(shape, dtype=df[value].dtype)  # Initialize the mask with the specified shape and dtype
    mask[ij[:, 0], ij[:, 1]] = df[value]  # Set the mask pixels at the coordinates to the corresponding values

    if selem is None:
        selem = np.ones((width, width))  # Use a square structuring element of the given width if selem is not provided
        
    mask = skimage.morphology.dilation(mask, selem)  # Perform dilation using the structuring element

    return mask  # Return the resulting mask

def add_base_codes(df_reads, bases, offset, col):
    """
    Add base codes to a DataFrame of reads.

    Args:
    df_reads (pandas.DataFrame): DataFrame containing the reads data.
    bases (str): String of bases to extract from each read in the DataFrame.
    offset (int): Offset value to add to the base codes.
    col (str): Column name containing the reads data.

    Returns:
    pandas.DataFrame: DataFrame with base codes added.

    Notes:
    - This function extracts base codes from a specified column in the DataFrame.
    - It applies an offset to the extracted base codes.
    - The modified DataFrame is returned with the base codes added.
    """
    n = len(df_reads[col].iloc[0])

    # Extract base codes from the specified column and map them to their indices
    df = (df_reads[col].str.extract('(.)'*n)
          .applymap(bases.index)
          .rename(columns=lambda x: 'c{0}'.format(x+1))
         )

    # Concatenate the extracted base codes with the original DataFrame and apply the offset
    return pd.concat([df_reads, df + offset], axis=1)


def annotate_bases(df_reads, col='barcode', bases='GTAC', offset=1, **kwargs):
    """
    Annotate bases for mapped and unmapped reads.

    Args:
    df_reads (pandas.DataFrame): DataFrame containing the reads data.
    col (str): Column name containing the reads data.
    bases (str): String of bases to extract from each read in the DataFrame.
    offset (int): Offset value to add to the base codes.
    **kwargs: Additional keyword arguments passed to the `annotate_points` function.

    Returns:
    numpy.ndarray: Array containing the annotated bases.

    Notes:
    - This function extracts base codes from a specified column in the DataFrame and annotates them.
    - It applies an offset to the extracted base codes.
    - The function returns an array containing the annotated bases for each cycle.
    """
    # Add base codes to the DataFrame
    df_reads = add_base_codes(df_reads, bases, offset, col)
    
    # Get the number of cycles
    n = len(df_reads[col].iloc[0])
    
    # Generate cycle labels
    cycles = ['c{0}'.format(i+1) for i in range(n)]
        
    # Annotate points for each cycle and store the results in an array
    labels = np.array([annotate_points(df_reads, c, **kwargs) for c in cycles])
    
    return labels

def relabel_array(arr, new_label_dict):
    """
    Map values in an integer array based on `new_label_dict`, a dictionary from
    old to new values.

    Parameters:
    arr (numpy.ndarray): The input integer array to be relabeled.
    new_label_dict (dict): A dictionary mapping old values to new values.

    Returns:
    numpy.ndarray: The relabeled integer array.

    Notes:
    - The function iterates through the items in `new_label_dict` and maps old values to new values in the array.
    - Values in the array that do not have a corresponding mapping in `new_label_dict` remain unchanged.
    """
    n = arr.max()  # Find the maximum value in the array
    arr_ = np.zeros(n + 1)  # Initialize an array to store the relabeled values
    for old_val, new_val in new_label_dict.items():
        if old_val <= n:  # Check if the old value is within the range of the array
            arr_[old_val] = new_val  # Map the old value to the new value in the relabeling array
    return arr_[arr]  # Return the relabeled array


def outline_mask(arr, direction='outer', width=1):
    """
    Remove interior of label mask in `arr`.

    Parameters:
    arr (numpy.ndarray): The input label mask array.
    direction (str, optional): The direction of outlining. 'outer' outlines the outer boundary, 'inner' outlines the inner boundary. Default is 'outer'.
    width (int, optional): The width of the structuring element used for erosion and dilation. Default is 1.

    Returns:
    numpy.ndarray: The label mask array with the outlined interior removed.

    Raises:
    ValueError: If `direction` is neither 'outer' nor 'inner'.
    """
    selem = skimage.morphology.disk(width)  # Create a disk-shaped structuring element with the specified width
    arr = arr.copy()  # Create a copy of the input array to avoid modifying the original array
    if direction == 'outer':  # If outlining direction is 'outer'
        mask = skimage.morphology.erosion(arr, selem)  # Erode the mask using the structuring element
        arr[mask > 0] = 0  # Set interior pixels to 0
        return arr  # Return the modified array
    elif direction == 'inner':  # If outlining direction is 'inner'
        mask1 = skimage.morphology.erosion(arr, selem) == arr  # Create a mask for pixels on the inner boundary
        mask2 = skimage.morphology.dilation(arr, selem) == arr  # Create a mask for pixels on the outer boundary
        arr[mask1 & mask2] = 0  # Set pixels within the inner boundary and outside the outer boundary to 0
        return arr  # Return the modified array
    else:  # If direction is neither 'outer' nor 'inner'
        raise ValueError(direction)  # Raise a ValueError
    

def bitmap_label(labels, positions, colors=None):
    """
    Create a bitmap image with labeled text at specified positions.

    Parameters:
    labels (list): List of strings representing labels.
    positions (list): List of tuples representing positions (row, column) for each label.
    colors (list, optional): List of colors for each label. Default is None.

    Returns:
    numpy.ndarray: Bitmap image with labeled text.

    Notes:
    - If `colors` is not provided, all labels are assigned the color 1.
    - The function iterates through each label, its position, and color (if provided), generates a bitmap image for the label text using `lasagna.io.bitmap_text`, and places it in the bitmap image at the specified position with the specified color.
    """
    positions = np.array(positions).astype(int)  # Convert positions to integers
    if colors is None:
        colors = [1] * len(labels)  # Assign default color 1 to all labels if colors are not provided
    i_all, j_all, c_all = [], [], []  # Initialize lists to store row indices, column indices, and colors
    for label, (i, j), color in zip(labels, positions, colors):  # Iterate through labels, positions, and colors
        if label == '':  # Skip empty labels
            continue
        i_px, j_px = np.where(lasagna.io.bitmap_text(label))  # Generate bitmap text for the label
        i_all += list(i_px + i)  # Append row indices adjusted by position to the list
        j_all += list(j_px + j)  # Append column indices adjusted by position to the list
        c_all += [color] * len(i_px)  # Append color to the list for each pixel
        
    shape = max(i_all) + 1, max(j_all) + 1  # Calculate the shape of the bitmap image
    arr = np.zeros(shape, dtype=int)  # Initialize the bitmap image array with zeros
    arr[i_all, j_all] = c_all  # Assign colors to pixels in the bitmap image based on indices
    return arr  # Return the bitmap image array


def build_discrete_lut(colors):
    """
    Build ImageJ lookup table for a list of discrete colors.

    Parameters:
    colors (list): List of discrete colors. Each color can be specified using various formats understood by `sns.color_palette`.

    Returns:
    numpy.ndarray: ImageJ lookup table (LUT) built from the provided colors.

    Notes:
    - If the values to label are in the range 0..N, N + 1 colors should be provided (zero value is usually black).
    - Color values should be understood by `sns.color_palette`, such as "blue", (1, 0, 0), or "#0000ff".
    """
    try:
        import seaborn as sns  # Try importing seaborn for color palette
        colors = sns.color_palette(colors)  # Get the color palette from seaborn
    except:
        pass  # Continue if seaborn is not available

    colors = 255 * np.array(colors)  # Scale the colors to the range 0-255

    # try to match ImageJ LUT rounding convention
    m = len(colors)  # Get the number of colors
    n = int(256 / m)  # Calculate the number of repeats for each color
    p = m - (256 - n * m)  # Calculate the number of colors with n + 1 repeats
    color_index_1 = list(np.repeat(range(0, p), n))  # Create indices for colors with n repeats
    color_index_2 = list(np.repeat(range(p, m), n + 1))  # Create indices for colors with n + 1 repeats
    color_index = color_index_1 + color_index_2  # Combine the indices
    return colors_to_imagej_lut(colors[color_index, :])  # Return the ImageJ lookup table built from the specified colors

def bitmap_draw_line(image, coords, width=1, dashed=False):
    """
    Draw a horizontal line on an image and return an image of the same shape.
    Optionally, draw a dashed line if requested.

    Parameters:
    image (numpy.ndarray): Input image array.
    coords (list of tuples): List of coordinate tuples [(x1, y1), (x2, y2)] representing line endpoints.
    width (int, optional): Width of the line. Defaults to 1.
    dashed (bool or list, optional): Whether to draw a dashed line. If True, uses default dash pattern [100, 50]. Defaults to False.

    Returns:
    numpy.ndarray: Image with the drawn line.

    Raises:
    ValueError: If drawing a dashed line between more than 2 points or drawing a dashed non-horizontal line.

    Notes:
    - The function supports drawing dashed lines. If dashed=True, it expects coords to have only two points.
    - The function handles images of different data types (uint16, uint8, binary) and sets appropriate mode and fill values accordingly.
    """
    import PIL.ImageDraw

    # Check if dashed line between more than 2 points
    if (len(coords) > 2) and (dashed is not False):
        raise ValueError('Drawing a dashed line between more than 2 points not supported.')

    # Check if dashed non-horizontal line
    if (coords[0][1] != coords[1][1]) and (dashed is not False):
        raise ValueError('Drawing a dashed non-horizontal line not supported')

    # Determine image mode and fill value based on dtype
    if image.dtype == np.uint16:
        mode = 'I;16'
        fill = 2 ** 16 - 1
    elif image.dtype == np.uint8:
        mode = 'L'
        fill = 2 ** 8 - 1
    else:
        mode = '1'
        fill = True

    # Create a new PIL image
    img = PIL.Image.new(mode, image.shape[:-3:-1])
    draw = PIL.ImageDraw.Draw(img, mode=mode)

    # Draw dashed line if requested
    if dashed:
        y = coords[0][1]
        if not isinstance(dashed, list):
            dashed = [100, 50]  # dash, gap
        xs = []
        x = coords[0][0]
        counter = count(start=0, step=1)
        while x < coords[1][0]:
            xs.append(x)
            c = next(counter)
            if c % 2 == 0:
                x += dashed[0]
            else:
                x += dashed[1]
        xs.append(coords[1][0])
        for x_0, x_1 in zip(xs[::2], xs[1::2]):
            draw.line([(x_0, y), (x_1, y)], width=width, fill=fill)
    else:
        draw.line(coords, width=width, fill=fill)

    return np.array(img)


def bitmap_text_overlay(image, anchor_point, text, size=10, font=VISITOR_FONT):
    """
    Draw text on an image in the shape of the given image.

    Parameters:
    image (numpy.ndarray): Input image array.
    anchor_point (tuple): Anchor point (x, y) where the text should be positioned.
    text (str): Text to be drawn on the image.
    size (int, optional): Font size. Defaults to 10.
    font (PIL.ImageFont.FreeTypeFont or str, optional): Font to be used for drawing the text.
        If a string is provided, the function loads the corresponding TrueType font.
        Defaults to VISITOR_FONT.

    Returns:
    numpy.ndarray: Image with the drawn text.

    Notes:
    - The function creates a new PIL image based on the input image shape.
    - It uses PIL's ImageDraw module to draw text on the image.
    - If the input image dtype is uint16, it converts the image to uint8 before drawing the text due to a PIL bug.
    - The font argument can be either a PIL.ImageFont.FreeTypeFont object or a string representing the font file path.
    """
    import PIL.ImageDraw

    # Determine image mode based on dtype
    if image.dtype == np.uint16:
        mode = 'L'  # PIL has a bug with drawing text on uint16 images
    elif image.dtype == np.uint8:
        mode = 'L'
    else:
        mode = '1'

    # Create a new PIL image
    img = PIL.Image.new(mode, image.shape[:-3:-1])
    draw = PIL.ImageDraw.Draw(img)

    # Load font if font argument is a string
    if isinstance(font, PIL.ImageFont.FreeTypeFont):
        FONT = font
        if FONT.size != size:
            warnings.warn(f'Size of supplied FreeTypeFont object is {FONT.size}, but input argument size = {size}.')
    else:
        FONT = load_truetype(truetype=font, size=size)
    offset = FONT.getoffset(text)

    # Draw text on the image
    draw.text(np.array(anchor_point) - np.array(offset), text, font=FONT, fill='white')

    # Convert image to uint16 if necessary
    if image.dtype == np.uint16:
        return skimage.img_as_uint(np.array(img))
    else:
        return np.array(img, dtype=image.dtype)

def bitmap_line(s, crop=True):
    """
    Draw text using Visitor font (characters are 5x5 pixels).

    Parameters:
    s (str): Text to be drawn.
    crop (bool, optional): Whether to crop the whitespace around the drawn text.
        Defaults to True.

    Returns:
    numpy.ndarray: Array representing the drawn text.

    Notes:
    - The function creates a new PIL RGBA image with a width determined by the length of the input text.
    - It draws the text on the image using the Visitor font.
    - It then extracts the alpha channel from the image, representing the drawn text.
    - If crop is True, it crops the whitespace around the drawn text.
    """
    import PIL.Image
    import PIL.ImageDraw

    # Create a new PIL RGBA image
    img = PIL.Image.new("RGBA", (len(s) * 8, 10), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)

    # Draw text on the image
    draw.text((0, 0), s, (255, 255, 255), font=VISITOR_FONT)

    # Get the alpha channel representing the drawn text
    n = np.array(img)[2:7, :, 0]

    # Crop the whitespace around the drawn text if requested
    if (n.sum() == 0) or (not crop):
        return n
    return (n[:, :np.where(n.any(axis=0))[0][-1] + 1] > 0).astype(int)


def bitmap_lines(lines, spacing=1, crop=True):
    """
    Draw multiple lines of text from a list of strings.

    Parameters:
    lines (list): List of strings representing lines of text to be drawn.
    spacing (int, optional): Spacing between lines. Defaults to 1.
    crop (bool, optional): Whether to crop the whitespace around the drawn text.
        Defaults to True.

    Returns:
    numpy.ndarray: Array representing the drawn text lines.

    Notes:
    - The function iterates over each line of text in the input list and calls `bitmap_line` to draw each line.
    - It calculates the height and maximum width of the drawn bitmaps.
    - It creates an output array to assemble the drawn text lines, ensuring proper spacing between lines.
    - If crop is True, it crops the whitespace around the drawn text lines.
    """
    # Draw each line of text and collect bitmaps
    bitmaps = [bitmap_line(x, crop=crop) for x in lines]
    height = 5
    shapes = np.array([x.shape for x in bitmaps])
    shape = (height + 1) * len(bitmaps), shapes[:, 1].max()

    # Assemble the drawn text lines into an output array
    output = np.zeros(shape, dtype=int)
    for i, bitmap in enumerate(bitmaps):
        start, end = i * (height + 1), (i + 1) * (height + 1) - 1
        output[start:end, :bitmap.shape[1]] = bitmap

    return output[:-1, :]

def colors_to_imagej_lut(lut_values):
    """
    Convert color values to an ImageJ lookup table (LUT) format.

    Parameters:
    lut_values (array-like): Color values to be converted. Each row represents a color,
        with three columns for red, green, and blue values.

    Returns:
    tuple: Tuple representing the ImageJ LUT format, with 256 red values followed by
        256 green values and then 256 blue values.

    Notes:
    - The function expects the input `lut_values` to be an array-like object, where each
        row contains three columns representing red, green, and blue values.
    - It transposes the input array to ensure that each color channel is arranged
        sequentially.
    - It flattens the transposed array and converts the values to integers before
        returning as a tuple.
    """
    # Transpose the input array to arrange color channels sequentially
    transposed_lut = np.array(lut_values).T
    # Flatten and convert the transposed array to integers
    flattened_lut = transposed_lut.flatten().astype(int)

    return tuple(flattened_lut)

def build_GRMC():
    """
    Build an ImageJ lookup table (LUT) with colors from seaborn dark palettes.

    Returns:
    tuple: Tuple representing the ImageJ LUT format, with 256 red values followed by
        256 green values and then 256 blue values.

    Notes:
    - The function imports seaborn library to access dark palettes for color selection.
    - It creates a list of colors and extends it with dark palettes for each color.
    - The resulting list is converted into a NumPy array and trimmed to retain only
        RGB values.
    - A zero-initialized array of shape (256, 3) is created to store the LUT.
    - The RGB values from the palette are scaled to the range 0-255 and assigned to the
        LUT array.
    - Finally, the LUT array is flattened and returned as a tuple in ImageJ LUT format.
    """
    import seaborn as sns

    # Define colors for the palette
    colors = (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)
    lut = []

    # Generate LUT from seaborn dark palettes for each color
    for color in colors:
        lut.append([0, 0, 0, 1])  # Add transparent color
        # Extend the lut list with dark palettes
        lut.extend(sns.dark_palette(color, n_colors=64 - 1))

    # Convert lut list to NumPy array and retain only RGB values
    lut = np.array(lut)[:, :3]

    # Initialize an array for the ImageJ LUT
    RGCM = np.zeros((256, 3), dtype=int)

    # Scale RGB values to the range 0-255 and assign to LUT array
    RGCM[:len(lut)] = (lut * 255).astype(int)

    # Flatten the LUT array and return as a tuple
    return tuple(RGCM.T.flatten())

def add_rect_bounds(df, width=10, ij='ij', bounds_col='bounds'):
    """
    Add rectangular bounds to a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    width (int, optional): Width of the rectangular bounds. Defaults to 10.
    ij (str or tuple, optional): Column name or tuple of column names representing the
        coordinates in the DataFrame. Defaults to 'ij'.
    bounds_col (str, optional): Name of the column to store the bounds. Defaults to 'bounds'.

    Returns:
    pandas.DataFrame: DataFrame with the rectangular bounds added.

    Notes:
    - This function computes rectangular bounds around coordinates in the DataFrame.
    - It iterates over the 'ij' column (or columns) to extract the coordinates.
    - For each coordinate, it computes the bounds as (i - width, j - width, i + width, j + width).
    - The bounds are stored in a list 'arr'.
    - The DataFrame is then assigned a new column named 'bounds_col' with the computed bounds.
    - The modified DataFrame is returned.
    """
    arr = []

    # Iterate over the DataFrame to compute rectangular bounds
    for i, j in df[list(ij)].values.astype(int):
        arr.append((i - width, j - width, i + width, j + width))

    # Assign the computed bounds to a new column in the DataFrame
    return df.assign(**{bounds_col: arr})


def make_sq_bounds(
        df,
        input_bounds=['bounds_0','bounds_1','bounds_2','bounds_3'],
        bounds_col='bounds'):
    """
    Make square bounds from given bounds in a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    input_bounds (list of str, optional): List of column names representing the input bounds.
        Defaults to ['bounds_0','bounds_1','bounds_2','bounds_3'].
    bounds_col (str, optional): Name of the column to store the square bounds.
        Defaults to 'bounds'.

    Returns:
    pandas.DataFrame: DataFrame with the square bounds added.

    Notes:
    - This function computes square bounds from the given input bounds.
    - It iterates over the DataFrame to extract the input bounds.
    - For each set of input bounds, it computes the width and height.
    - It then computes the difference between height and width to determine padding.
    - The bounds are adjusted to make them square with appropriate padding.
    - The modified DataFrame is returned with the square bounds added.
    """
    # Define a function to split padding into equal parts
    def split_pad(pad):
        return (pad // 2, pad // 2 + pad % 2)

    arr = []

    # Iterate over the DataFrame to compute square bounds
    for bounds in df[input_bounds].values.astype(int):
        width, height = (bounds[2] - bounds[0]), (bounds[3] - bounds[1])
        diff = height - width
        pad_width, pad_height = split_pad(np.clip(diff, 0, None)), split_pad(np.clip(-diff, 0, None))
        arr.append(tuple(bounds + np.array([-pad_width[0], -pad_height[0], pad_width[1], pad_height[1]])))

    # Assign the computed square bounds to a new column in the DataFrame
    return df.assign(**{bounds_col: arr})

# Define the colors used for base labeling
colors = (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)

# Build the discrete lookup table for base labeling
GRMC = build_discrete_lut(colors)