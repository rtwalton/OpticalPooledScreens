import warnings
from collections import defaultdict
from collections.abc import Iterable
from itertools import product

import skimage
import skimage.registration
import skimage.segmentation
import skimage.feature
import skimage.filters
import numpy as np
import pandas as pd
import scipy.stats

from scipy import ndimage

import ops.io
import ops.utils

# FEATURES
def feature_table(data, labels, features, global_features=None):
    """
    Apply functions in feature dictionary to regions in data specified by integer labels.
    If provided, the global feature dictionary is applied to the full input data and labels.

    Results are combined in a dataframe with one row per label and one column per feature.
    
    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = utils.regionprops(labels, intensity_image=data)
    
    # Initialize a defaultdict to store feature values
    results = defaultdict(list)
    
    # Loop through each region and compute features
    for region in regions:
        for feature, func in features.items():
            # Apply the feature function to the region and append the result to the corresponding feature list
            results[feature].append(fix_uint16(func(region)))
    
    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = fix_uint16(func(data, labels))

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)

def feature_table_multichannel(data, labels, features, global_features=None):
    """
    Apply functions in feature dictionary to regions in data specified by integer labels.
    If provided, the global feature dictionary is applied to the full input data and labels.

    Results are combined in a dataframe with one row per label and one column per feature.
    
    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = utils.regionprops_multichannel(labels, intensity_image=data)
    
    # Initialize a defaultdict to store feature values
    results = defaultdict(list)
    
    # Loop through each feature and compute features for each region
    for feature, func in features.items():
        # Check if the result of applying the function to the first region is iterable
        result_0 = func(regions[0])
        if isinstance(result_0, Iterable):
            if len(result_0) == 1:
                # If the result is a single value, apply the function to each region and append the result to the corresponding feature list
                results[feature] = [func(region)[0] for region in regions]
            else:
                # If the result is a sequence, apply the function to each region and append each element of the result to the corresponding feature list
                for result in map(func, regions):
                    for index, value in enumerate(result):
                        results[f"{feature}_{index}"].append(value)
        else:
            # If the result is not iterable, apply the function to each region and append the result to the corresponding feature list
            results[feature] = list(map(func, regions))

    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = func(data, labels)
    
    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


def fix_uint16(x):
    """
    Pandas bug converts np.uint16 to np.int16!!! 
    
    Args:
        x (Union[np.uint16, int]): Value to fix.
    
    Returns:
        Union[int, np.uint16]: Fixed value.
    """
    if isinstance(x, np.uint16):
        return int(x)
    return x


def build_feature_table(stack, labels, features, index):
    """Iterate over leading dimensions of stack, applying `feature_table`. 
    Results are labeled by index and concatenated.

        >>> stack.shape 
        (3, 4, 511, 626)
        
        index = (('round', range(1,4)), 
                 ('channel', ('DAPI', 'Cy3', 'A594', 'Cy5')))
    
        build_feature_table(stack, labels, features, index) 

    """
    index_vals = list(product(*[vals for _, vals in index]))
    index_names = [x[0] for x in index]
    
    s = stack.shape
    results = []
    for frame, vals in zip(stack.reshape(-1, s[-2], s[-1]), index_vals):
        df = feature_table(frame, labels, features)
        for name, val in zip(index_names, vals):
            df[name] = val
        results += [df]
    
    return pd.concat(results)


def find_cells(nuclei, mask, remove_boundary_cells=True):
    """
    Convert binary mask to cell labels, based on nuclei labels.

    Expands labeled nuclei to cells, constrained to where mask is >0.

    Parameters:
        nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
        mask (numpy.ndarray): Binary mask indicating valid regions for cell expansion.
        remove_boundary_cells (bool, optional): Whether to remove cells touching the boundary. Default is True.

    Returns:
        numpy.ndarray: Labeled segmentation mask of cells.
    """
    # Calculate distance transform of areas where nuclei are not present
    distance = ndi.distance_transform_cdt(nuclei == 0)
    
    # Use watershed segmentation to expand nuclei labels to cells within the mask
    cells = skimage.segmentation.watershed(distance, nuclei, mask=mask)
    
    # Remove cells touching the boundary if specified
    if remove_boundary_cells:
        # Identify cells touching the boundary
        cut = np.concatenate([cells[0, :], cells[-1, :], cells[:, 0], cells[:, -1]])
        # Set labels of boundary-touching cells to 0
        cells.flat[np.in1d(cells, np.unique(cut))] = 0

    return cells.astype(np.uint16)


def find_peaks(data, n=5):
    """
    Finds local maxima in the input data.
    At a maximum, the value is max - min in a neighborhood of width `n`.
    Elsewhere, it is zero.

    Parameters:
        data (numpy.ndarray): Input data.
        n (int, optional): Width of the neighborhood for finding local maxima. Default is 5.

    Returns:
        peaks (numpy.ndarray): Local maxima scores.
    """
    # Import necessary modules and functions
    from scipy import ndimage as ndi
    import numpy as np
    
    # Define the maximum and minimum filters for finding local maxima
    filters = ndi.filters
    
    # Define the neighborhood size based on the input data dimensions
    neighborhood_size = (1,) * (data.ndim - 2) + (n, n)
    
    # Apply maximum and minimum filters to the data to find local maxima
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    
    # Calculate the difference between maximum and minimum values to identify peaks
    peaks = data_max - data_min
    
    # Set values to zero where the original data is not equal to the maximum values
    peaks[data != data_max] = 0
    
    # Remove peaks close to the edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0
    
    return peaks

def calculate_illumination_correction(files, smooth=None, rescale=True, threading=False, slicer=slice(None)):
    """
    Calculate illumination correction field for use with the apply_illumination_correction
    Snake method. Equivalent to CellProfiler's CorrectIlluminationCalculate module with
    option "Regular", "All", "Median Filter".

    Note: Algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
    illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014).

    Parameters:
    -----------
    files : list
        List of file paths to images for which to calculate the illumination correction.
    smooth : int, optional
        Smoothing factor for the correction. Default is calculated as 1/20th of the image area.
    rescale : bool, default True
        Whether to rescale the correction field.
    threading : bool, default False
        Whether to use threading for parallel processing.
    slicer : slice, optional
        Slice object to select specific parts of the images.

    Returns:
    --------
    numpy.ndarray
        The calculated illumination correction field.
    """
    from ops.io import read_stack as read
    from joblib import Parallel, delayed
    import numpy as np
    import skimage.morphology
    import skimage.filters
    import warnings
    import ops.utils

    N = len(files)

    # Initialize global data variable
    global data
    data = read(files[0])[slicer] / N

    def accumulate_image(file):
        global data
        data += read(file)[slicer] / N

    # Accumulate images using threading or sequential processing
    if threading:
        Parallel(n_jobs=-1, require='sharedmem')(delayed(accumulate_image)(file) for file in files[1:])
    else:
        for file in files[1:]:
            accumulate_image(file)

    # Squeeze and convert data to uint16
    data = np.squeeze(data.astype(np.uint16))

    # Calculate default smoothing factor if not provided
    if not smooth:
        smooth = int(np.sqrt((data.shape[-1] * data.shape[-2]) / (np.pi * 20)))

    selem = skimage.morphology.disk(smooth)
    median_filter = ops.utils.applyIJ(skimage.filters.median)

    # Apply median filter with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = median_filter(data, selem, behavior='rank')

    # Rescale channels if requested
    if rescale:
        @ops.utils.applyIJ
        def rescale_channels(data):
            # Use 2nd percentile for robust minimum
            robust_min = np.quantile(data.reshape(-1), q=0.02)
            robust_min = 1 if robust_min == 0 else robust_min
            data = data / robust_min
            data[data < 1] = 1
            return data

        smoothed = rescale_channels(smoothed)

    return smoothed

@ops.utils.applyIJ
def rolling_ball_background_skimage(image, radius=100, ball=None, shrink_factor=None, smooth=None, **kwargs):
    """
    Apply rolling ball background subtraction to an image using skimage.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image for background subtraction.
    radius : int, default 100
        Radius of the rolling ball.
    ball : numpy.ndarray, optional
        Precomputed ball kernel. If None, it will be generated.
    shrink_factor : int, optional
        Factor by which to shrink the image and ball for faster computation. 
        Default is determined based on the radius.
    smooth : float, optional
        Sigma for Gaussian smoothing applied to the background after rolling ball.
    kwargs : dict
        Additional arguments passed to skimage's rolling_ball function.

    Returns:
    --------
    numpy.ndarray
        The calculated background to be subtracted from the original image.
    """
    import skimage.restoration
    import skimage.transform
    import skimage.filters

    # Generate the ball kernel if not provided
    if ball is None:
        ball = skimage.restoration.ball_kernel(radius, ndim=2)

    # Determine shrink factor and trim based on the radius
    if shrink_factor is None:
        if radius <= 10:
            shrink_factor = 1
            trim = 0.12  # Trim 24% in x and y
        elif radius <= 30:
            shrink_factor = 2
            trim = 0.12  # Trim 24% in x and y
        elif radius <= 100:
            shrink_factor = 4
            trim = 0.16  # Trim 32% in x and y
        else:
            shrink_factor = 8
            trim = 0.20  # Trim 40% in x and y

        # Trim the ball kernel
        n = int(ball.shape[0] * trim)
        i0, i1 = n, ball.shape[0] - n
        ball = ball[i0:i1, i0:i1]

    # Rescale the image and ball kernel
    image_rescaled = skimage.transform.rescale(image, 1.0 / shrink_factor, preserve_range=True).astype(image.dtype)
    kernel_rescaled = skimage.transform.rescale(ball, 1.0 / shrink_factor, preserve_range=True).astype(ball.dtype)

    # Compute the rolling ball background
    background = skimage.restoration.rolling_ball(image_rescaled, kernel=kernel_rescaled, **kwargs)

    # Apply Gaussian smoothing if specified
    if smooth is not None:
        background = skimage.filters.gaussian(background, sigma=smooth / shrink_factor, preserve_range=True)

    # Resize the background to the original image size
    background_resized = skimage.transform.resize(background, image.shape, preserve_range=True).astype(image.dtype)

    return background_resized

def subtract_background(image, radius=100, ball=None, shrink_factor=None, smooth=None, **kwargs):
    """
    Subtract the background from an image using the rolling ball algorithm.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image from which to subtract the background.
    radius : int, default 100
        Radius of the rolling ball.
    ball : numpy.ndarray, optional
        Precomputed ball kernel. If None, it will be generated.
    shrink_factor : int, optional
        Factor by which to shrink the image and ball for faster computation. 
        Default is determined based on the radius.
    smooth : float, optional
        Sigma for Gaussian smoothing applied to the background after rolling ball.
    kwargs : dict
        Additional arguments passed to the rolling_ball_background_skimage function.

    Returns:
    --------
    numpy.ndarray
        The image with the background subtracted.
    """
    # Calculate the background using the rolling ball algorithm
    background = rolling_ball_background_skimage(image, radius=radius, ball=ball,
                                                 shrink_factor=shrink_factor, smooth=smooth, **kwargs)

    # Ensure that the background does not exceed the image values
    mask = background > image
    background[mask] = image[mask]

    # Subtract the background from the image
    return image - background

@ops.utils.applyIJ
def log_ndi(data, sigma=1, *args, **kwargs):
    """
    Apply Laplacian of Gaussian to each image in a stack of shape (..., I, J).
    
    Parameters:
        data (numpy.ndarray): Input data.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Default is 1.
        *args: Additional positional arguments passed to scipy.ndimage.filters.gaussian_laplace.
        **kwargs: Additional keyword arguments passed to scipy.ndimage.filters.gaussian_laplace.
    
    Returns:
        numpy.ndarray: Resulting images after applying Laplacian of Gaussian.
    """
    # Define the Laplacian of Gaussian filter function
    f = ndi.filters.gaussian_laplace
    
    # Apply the filter to the data and invert the output
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
    
    # Clip values to ensure they are within the valid range [0, 65535] and convert back to uint16
    arr_ = np.clip(arr_, 0, 65535) / 65535
    
    # Suppress precision warning from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)


class Align:
    """Alignment redux, used by snakemake.
    """
    @staticmethod
    def normalize_by_percentile(data_, q_norm=70):
        """Normalize data by the specified percentile.

        Parameters
        ----------
        data_ : numpy array
            Input image data.
        q_norm : int, optional, default: 70
            Percentile value for normalization.

        Returns
        -------
        normed : numpy array
            Normalized image data.
        """
        # Get the shape of the input data
        shape = data_.shape
        # Replace the last two dimensions with a single dimension to allow percentile calculation
        shape = shape[:-2] + (-1,)
        # Calculate the q_normth percentile along the last two dimensions of the data
        p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
        # Normalize the data by dividing it by the calculated percentile values
        normed = data_ / p
        # Return the normalized data
        return normed

    @staticmethod
    @ops.utils.applyIJ
    def filter_percentiles(data, q1, q2):
        """Replace data outside of the percentile range [q1, q2] with uniform noise.

        Parameters
        ----------
        data : numpy array
            Input image data.
        q1 : int
            Lower percentile threshold.
        q2 : int
            Upper percentile threshold.

        Returns
        -------
        filtered : numpy array
            Filtered image data.
        """
        # Calculate the q1th and q2th percentiles of the input data
        x1, x2 = np.percentile(data, [q1, q2])
        # Create a mask where values are outside the range [x1, x2]
        mask = (x1 > data) | (x2 < data)
        # Fill the masked values with uniform noise in the range [x1, x2] using the fill_noise function
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    @ops.utils.applyIJ
    def filter_values(data, x1, x2):
        """Replace data outside of the value range [x1, x2] with uniform noise.

        Parameters
        ----------
        data : numpy array
            Input image data.
        x1 : int
            Lower value threshold.
        x2 : int
            Upper value threshold.

        Returns
        -------
        filtered : numpy array
            Filtered image data.
        """
        # Create a mask where values are either less than x1 or greater than x2
        mask = (x1 > data) | (x2 < data)
        # Fill the masked values with uniform noise in the range [x1, x2] using the fill_noise function
        return Align.fill_noise(data, mask, x1, x2)


    @staticmethod
    def fill_noise(data, mask, x1, x2):
        """Fill masked areas of data with uniform noise.

        Parameters
        ----------
        data : numpy array
            Input image data.
        mask : numpy array
            Boolean mask indicating areas to be replaced with noise.
        x1 : int
            Lower threshold value.
        x2 : int
            Upper threshold value.

        Returns
        -------
        filtered : numpy array
            Filtered image data.
        """
         # Make a copy of the original data
        filtered = data.copy()
        # Initialize a random number generator with seed 0
        rs = np.random.RandomState(0)
        # Replace the masked values with uniform noise generated in the range [x1, x2]
        filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
        # Return the filtered data
        return filtered

    @staticmethod
    def calculate_offsets(data_, upsample_factor):
        """Calculate offsets between images using phase cross-correlation.

        Parameters
        ----------
        data_ : numpy array
            Image data.
        upsample_factor : int
            Upsampling factor for cross-correlation.

        Returns
        -------
        offsets : numpy array
            Offset values between images.
        """
        # Set the target frame as the first frame in the data
        target = data_[0]
        # Initialize an empty list to store offsets
        offsets = []
        # Iterate through each frame in the data
        for i, src in enumerate(data_):
            # If it's the first frame, add a zero offset
            if i == 0:
                offsets += [(0, 0)]
            else:
                # Calculate the offset between the current frame and the target frame
                offset, _, _ = skimage.registration.phase_cross_correlation(
                                src, target, upsample_factor=upsample_factor)
                # Add the offset to the list
                offsets += [offset]
        # Convert the list of offsets to a numpy array and return
        return np.array(offsets)

    @staticmethod
    def apply_offsets(data_, offsets):
        """Apply offsets to image data.

        Parameters
        ----------
        data_ : numpy array
            Image data.
        offsets : numpy array
            Offset values to be applied.

        Returns
        -------
        warped : numpy array
            Warped image data.
        """
        # Initialize an empty list to store warped frames
        warped = []
        # Iterate through each frame and its corresponding offset
        for frame, offset in zip(data_, offsets):
            # If the offset is zero, add the frame as it is
            if offset[0] == 0 and offset[1] == 0:
                warped += [frame]
            else:
                # Otherwise, apply a similarity transform to warp the frame based on the offset
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_ = skimage.transform.warp(frame, st, preserve_range=True)
                # Add the warped frame to the list
                warped += [frame_.astype(data_.dtype)]
        # Convert the list of warped frames to a numpy array and return
        return np.array(warped)

    @staticmethod
    def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
        """Align images within the same cycle.

        Parameters
        ----------
        data_ : numpy array
            Image data.
        upsample_factor : int, optional, default: 4
            Upsampling factor for cross-correlation.
        window : int, optional, default: 1
            Size of the window to apply during alignment.
        q1 : int, optional, default: 0
            Lower percentile threshold.
        q2 : int, optional, default: 90
            Upper percentile threshold.

        Returns
        -------
        aligned : numpy array
            Aligned image data.
        """
        # Filter the input data based on percentiles
        filtered = Align.filter_percentiles(Align.apply_window(data_, window), q1=q1, q2=q2)
        # Calculate offsets using the filtered data
        offsets = Align.calculate_offsets(filtered, upsample_factor=upsample_factor)
        # Apply the calculated offsets to the original data and return the result
        return Align.apply_offsets(data_, offsets)

    @staticmethod
    def align_between_cycles(data, channel_index, upsample_factor=4, window=1, return_offsets=False):
        """Align images between different cycles.

        Parameters
        ----------
        data : numpy array
            Image data.
        channel_index : int
            Index of the channel to align between cycles.
        upsample_factor : int, optional, default: 4
            Upsampling factor for cross-correlation.
        window : int, optional, default: 1
            Size of the window to apply during alignment.
        return_offsets : bool, optional, default: False
            Whether to return the calculated offsets.

        Returns
        -------
        aligned : numpy array
            Aligned image data.
        offsets : numpy array, optional
            Calculated offsets if return_offsets is True.
        """
        # Calculate offsets from the target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor)

        # Apply the calculated offsets to all channels
        warped = []
        for data_ in data.transpose([1, 0, 2, 3]):
            warped += [Align.apply_offsets(data_, offsets)]

        # Transpose the array back to its original shape
        aligned = np.array(warped).transpose([1, 0, 2, 3])

        # Return aligned data with offsets if requested
        if return_offsets:
            return aligned, offsets
        else:
            return aligned


    @staticmethod
    def apply_window(data, window):
        """Apply a window to image data.

        Parameters
        ----------
        data : numpy array
            Image data.
        window : int
            Size of the window to apply.

        Returns
        -------
        filtered : numpy array
            Filtered image data.
        """
        # Extract height and width dimensions from the last two axes of the data shape
        height, width = data.shape[-2:]

        # Define a function to find the border based on the window size
        find_border = lambda x: int((x/2.) * (1 - 1/float(window)))

        # Calculate border indices
        i, j = find_border(height), find_border(width)

        # Return the data with the border cropped out
        return data[..., i:height - i, j:width - j]



def find_nuclei(dapi, threshold, radius=15, area_min=50, area_max=500,
                score=lambda r: r.mean_intensity,
                smooth=1.35):
    """
    Segment nuclei from DAPI stain using various parameters and filters.

    Parameters:
        dapi (numpy.ndarray): Input DAPI image.
        threshold (float): Threshold for mean intensity to segment nuclei.
        radius (int, optional): Radius of disk used in local mean thresholding to identify foreground. Default is 15.
        area_min (int, optional): Minimum area for retaining nuclei after segmentation. Default is 50.
        area_max (int, optional): Maximum area for retaining nuclei after segmentation. Default is 500.
        score (function, optional): Function to calculate region score. Default is lambda r: r.mean_intensity.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map to foreground prior to watershedding. Default is 1.35.

    Returns:
        result (numpy.ndarray): Labeled segmentation mask of nuclei.
    """

    # Binarize DAPI image to identify foreground
    mask = binarize(dapi, radius, area_min)
    
    # Label connected components in the binary mask
    labeled = skimage.measure.label(mask)
    
    # Filter labeled regions based on intensity score and threshold
    labeled = filter_by_region(labeled, score, threshold, intensity_image=dapi) > 0

    # Fill holes in the labeled mask
    filled = ndi.binary_fill_holes(labeled)
    
    # Label the differences between filled and original labeled regions
    difference = skimage.measure.label(filled != labeled)

    # Identify regions with changes in area and update labeled mask
    change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    # Apply watershed algorithm to refine segmentation
    nuclei = apply_watershed(labeled, smooth=smooth)

    # Filter resulting nuclei by area range
    result = filter_by_region(nuclei, lambda r: area_min < r.area < area_max, threshold)

    return result


def find_foci(data, radius=3, threshold=10, remove_border_foci=False):
    tophat = skimage.morphology.white_tophat(data,selem=skimage.morphology.disk(radius))
    print("finding foci")
    tophat_log = log_ndi(tophat, sigma=radius)

    mask = tophat_log > threshold
    mask = skimage.morphology.remove_small_objects(mask,min_size=(radius**2))
    labeled = skimage.measure.label(mask)
    labeled = apply_watershed(labeled,smooth=1)

    if remove_border_foci:
        border_mask = data>0
        labeled = remove_border(labeled,~border_mask)

    return labeled

def remove_border(labels, mask, dilate=5):
    mask = skimage.morphology.binary_dilation(mask,np.ones((dilate,dilate)))
    remove = np.unique(labels[mask])
    labels = labels.copy()
    labels.flat[np.in1d(labels,remove)] = 0
    return labels

def binarize(image, radius, min_size):
    """Apply local mean threshold to find outlines. Filter out
    background shapes. Otsu threshold on list of region mean intensities will remove a few
    dark cells. Could use shape to improve the filtering.
    """
    dapi = skimage.img_as_ubyte(image)
    # slower than optimized disk in ImageJ
    # scipy.ndimage.uniform_filter with square is fast but crappy
    selem = skimage.morphology.disk(radius)
    mean_filtered = skimage.filters.rank.mean(dapi, selem=selem)
    mask = dapi > mean_filtered
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    return mask


def filter_by_region(labeled, score, threshold, intensity_image=None, relabel=True):
    """Apply a filter to label image. The `score` function takes a single region 
    as input and returns a score. 
    If scores are boolean, regions where the score is false are removed.
    Otherwise, the function `threshold` is applied to the list of scores to 
    determine the minimum score at which a region is kept.
    If `relabel` is true, the regions are relabeled starting from 1.
    """
    labeled = labeled.copy().astype(int)
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image)
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        t = threshold(scores)
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0
    
    if relabel:
        labeled, _, _ = skimage.segmentation.relabel_sequential(labeled)

    return labeled


def apply_watershed(img, smooth=4):
    distance = ndi.distance_transform_edt(img)
    if smooth > 0:
        distance = skimage.filters.gaussian(distance, sigma=smooth)
    local_max = skimage.feature.peak_local_max(
                    distance, indices=False, footprint=np.ones((3, 3)), 
                    exclude_border=False)

    markers = ndi.label(local_max)[0]
    result = skimage.segmentation.watershed(-distance, markers, mask=img)
    return result.astype(np.uint16)


def alpha_blend(arr, positions, clip=True, edge=0.95, edge_width=0.02, subpixel=False):
    """Blend array of images, translating image coordinates according to offset matrix.
    arr : N x I x J
    positions : N x 2 (n, i, j)
    """
    
    # @utils.memoize
    def make_alpha(s, edge=0.95, edge_width=0.02):
        """Unity in center, drops off near edge
        :param s: shape
        :param edge: mid-point of drop-off
        :param edge_width: width of drop-off in exponential
        :return:
        """
        sigmoid = lambda r: 1. / (1. + np.exp(-r))

        x, y = np.meshgrid(range(s[0]), range(s[1]))
        xy = np.concatenate([x[None, ...] - s[0] / 2,
                             y[None, ...] - s[1] / 2])
        R = np.max(np.abs(xy), axis=0)

        return sigmoid(-(R - s[0] * edge/2) / (s[0] * edge_width))

    # determine output shape, offset positions as necessary
    if subpixel:
        positions = np.array(positions)
    else:
        positions = np.round(positions)
    # convert from ij to xy
    positions = positions[:, [1, 0]]    

    positions -= positions.min(axis=0)
    shapes = [a.shape for a in arr]
    output_shape = np.ceil((shapes + positions[:,::-1]).max(axis=0)).astype(int)

    # sum data and alpha layer separately, divide data by alpha
    output = np.zeros([2] + list(output_shape), dtype=float)
    for image, xy in zip(arr, positions):
        alpha = 100 * make_alpha(image.shape, edge=edge, edge_width=edge_width)
        if subpixel is False:
            j, i = np.round(xy).astype(int)

            output[0, i:i+image.shape[0], j:j+image.shape[1]] += image * alpha.T
            output[1, i:i+image.shape[0], j:j+image.shape[1]] += alpha.T
        else:
            ST = skimage.transform.SimilarityTransform(translation=xy)

            tmp = np.array([skimage.transform.warp(image, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='reflect'),
                            skimage.transform.warp(alpha, inverse_map=ST.inverse,
                                                   output_shape=output_shape,
                                                   preserve_range=True, mode='constant')])
            tmp[0, :, :] *= tmp[1, :, :]
            output += tmp


    output = (output[0, :, :] / output[1, :, :])

    if clip:
        def edges(n):
            return np.r_[n[:4, :].flatten(), n[-4:, :].flatten(),
                         n[:, :4].flatten(), n[:, -4:].flatten()]

        while np.isnan(edges(output)).any():
            output = output[4:-4, 4:-4]

    return output.astype(arr[0].dtype)