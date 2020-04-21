
import mahotas 
import numpy as np
from astropy.stats import median_absolute_deviation
from ops.features import correlate_channels_masked, masked

def masked_rect(r,index):
    #masked image, maintaing shape
    image = r.intensity_image_full[index]
    mask = r.filled_image
    return np.multiply(image,mask)

def mahotas_zernike(r,channel):
    image = masked_rect(r,channel)
    mfeat = mahotas.features.zernike_moments(image.astype(np.uint32), radius = 9, degree=9)
    return mfeat

def mahotas_pftas(r,channel):
    image = masked_rect(r,channel)
    mfeat = mahotas.features.pftas(image.astype(np.uint32))
    ### according to this, at least as good as haralick/zernike and much faster:
    ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
    return mfeat

features_nuclear = {
    'dapi_nuclear_min': lambda r: np.min(masked(r,0)),
    'dapi_nuclear_25': lambda r: np.percentile(masked(r, 0),25),
    'dapi_nuclear_mean': lambda r: masked(r, 0).mean(),
    'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
    'dapi_nuclear_75': lambda r: np.percentile(masked(r, 0),75),
    'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
    'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
    'dapi_nuclear_sd': lambda r: np.std(masked(r,0)),
    'dapi_nuclear_mad': lambda r: median_absolute_deviation(masked(r,0)),
    # 'dapi_zernike_nuclear': lambda r: mahotas_zernike(r,0),
    # 'dapi_pftas_nuclear': lambda r: mahotas_pftas(r,0),
    'dm1a_nuclear_min': lambda r: np.min(masked(r,1)),
    'dm1a_nuclear_25': lambda r: np.percentile(masked(r, 1),25),
    'dm1a_nuclear_mean' : lambda r: masked(r, 1).mean(),
    'dm1a_nuclear_median' : lambda r: np.median(masked(r, 1)),
    'dm1a_nuclear_75': lambda r: np.percentile(masked(r, 1),75),
    'dm1a_nuclear_max'    : lambda r: masked(r, 1).max(),
    'dm1a_nuclear_int'    : lambda r: masked(r, 1).sum(),
    'dm1a_nuclear_sd': lambda r: np.std(masked(r,1)),
    'dm1a_nuclear_mad': lambda r: median_absolute_deviation(masked(r,1)),
    #
    'gh2ax_nuclear_min': lambda r: np.min(masked(r,2)),
    'gh2ax_nuclear_25': lambda r: np.percentile(masked(r, 2),25),
    'gh2ax_nuclear_mean' : lambda r: masked(r, 2).mean(),
    'gh2ax_nuclear_median' : lambda r: np.median(masked(r, 2)),
    'gh2ax_nuclear_75': lambda r: np.percentile(masked(r, 2),75),
    'gh2ax_nuclear_max'    : lambda r: masked(r, 2).max(),
    'gh2ax_nuclear_int'    : lambda r: masked(r, 2).sum(),
    'gh2ax_nuclear_sd': lambda r: np.std(masked(r,2)),
    'gh2ax_nuclear_mad': lambda r: median_absolute_deviation(masked(r,2)),
    #
    'phalloidin_nuclear_min': lambda r: np.min(masked(r,3)),
    'phalloidin_nuclear_25': lambda r: np.percentile(masked(r, 3),25),
    'phalloidin_nuclear_mean' : lambda r: masked(r, 3).mean(),
    'phalloidin_nuclear_median' : lambda r: np.median(masked(r, 3)),
    'phalloidin_nuclear_75': lambda r: np.percentile(masked(r, 3),75),
    'phalloidin_nuclear_max'    : lambda r: masked(r, 3).max(),
    'phalloidin_nuclear_int'    : lambda r: masked(r, 3).sum(),
    'phalloidin_nuclear_sd': lambda r: np.std(masked(r,3)),
    'phalloidin_nuclear_mad': lambda r: median_absolute_deviation(masked(r,3)),     
    # 'dm1a_zernike_nuclear': lambda r: mahotas_zernike(r,1),
    # 'dm1a_pftas_nuclear': lambda r: mahotas_pftas(r,1),
    'dapi_dm1a_corr_nuclear': lambda r: correlate_channels_masked(r,0,1),
    'dapi_gh2ax_corr_nuclear': lambda r: correlate_channels_masked(r,0,2),
    'dm1a_gh2ax_corr_nuclear': lambda r: correlate_channels_masked(r,1,2),
    'dapi_phalloidin_corr_nuclear': lambda r: correlate_channels_masked(r,0,3),
    'dm1a_phalloidin_corr_nuclear': lambda r: correlate_channels_masked(r,1,3),
    'gh2ax_phalloidin_corr_nuclear': lambda r: correlate_channels_masked(r,2,3),
    # 'area_nuclear'       : lambda r: r.area,
    'perimeter_nuclear' : lambda r: r.perimeter,
    'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
    'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
    'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
    'orientation_nuclear' : lambda r: r.orientation,
    # 'hu_moments_nuclear': lambda r: r.moments_hu,
    'solidity_nuclear': lambda r: r.solidity,
    'extent_nuclear': lambda r: r.extent,
    # 'nucleus'               : lambda r: r.label,
    # 'i'       : lambda r: r.centroid[0],
    # 'j'       : lambda r: r.centroid[1]
}


features_cell = {
    'channel_cell_min': lambda r: np.min(masked(r,1)),
    'channel_cell_25': lambda r: np.percentile(masked(r, 1),25),
    'channel_cell_mean' : lambda r: masked(r, 1).mean(),
    'channel_cell_median' : lambda r: np.median(masked(r, 1)),
    'channel_cell_75': lambda r: np.percentile(masked(r, 1),75),
    'channel_cell_max'    : lambda r: masked(r, 1).max(),
    'channel_cell_int'    : lambda r: masked(r, 1).sum(),
    'channel_cell_sd': lambda r: np.std(masked(r,1)),
    'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,1)),    
    'channel_zernike_cell': lambda r: mahotas_zernike(r,1),
    'channel_pftas_cell': lambda r: mahotas_pftas(r,1),
    'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
    'dapi_channel_corr_cell': lambda r: correlate_channels_masked(r,0,1),
    'area_cell'       : lambda r: r.area,
    'perimeter_cell' : lambda r: r.perimeter,
    'euler_cell' : lambda r: r.euler_number,
    'eccentricity_cell' : lambda r: r.eccentricity, #cell
    'major_axis_cell' : lambda r: r.major_axis_length, #cell
    # 'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
    'orientation_cell' : lambda r: r.orientation,
    'hu_moments_cell': lambda r: r.moments_hu,
    'solidity_cell': lambda r: r.solidity,
    'extent_cell': lambda r: r.extent,
    'cell'               : lambda r: r.label,
    'bounds': lambda r: r.bbox
    #Luke:add more dapi cell features?
}
