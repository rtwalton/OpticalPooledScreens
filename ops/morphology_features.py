
import mahotas 
from astropy.stats import median_absolute_deviation
from ops.features import correlate_channels, masked

def mahotas_zernike(region):
    mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
    return mfeat

def mahotas_pftas(region):
    mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
    ### according to this, at least as good as haralick/zernike and much faster:
    ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
    return mfeat

features_nuclear = {
    'channel_nuclear_min': lambda r: np.min(masked(r,0)),
    'dapi_nuclear_25': lambda r: np.percentile(masked(r, 0),25),
    'dapi_nuclear_mean': lambda r: masked(r, 0).mean(),
    'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
    'dapi_nuclear_75': lambda r: np.percentile(masked(r, 0),75),
    'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
    'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
    'dapi_nuclear_sd': lambda r: np.std(masked(r,channel)),
    'dapi_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),
    'dapi_zernike_nuclear': lambda r: mahotas_zernike(masked(r,0)),
    'dapi_pftas_nuclear': lambda r: mahotas_pftas(masked(r,0)),
    'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
    'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
    'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
    'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
    'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
    'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
    'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
    'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
    'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),     
    'channel_zernike_nuclear': lambda r: mahotas_zernike(masked(r,channel)),
    'channel_pftas_nuclear': lambda r: mahotas_pftas(masked(r,channel)),
    'dapi_channel_corr_nuclear': lambda r: correlate_channels(np.array([masked(r,i) for i in range(2)]),0,1),
    'area_nuclear'       : lambda r: r.area,
    'perimeter_nuclear' : lambda r: r.perimeter,
    'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
    'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
    'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
    'orientation_nuclear' : lambda r: r.orientation,
    'hu_moments_nuclear': lambda r: r.moments_hu,
    'solidity_nuclear': lambda r: r.solidity,
    'extent_nuclear': lambda r: r.extent,
    'cell'               : lambda r: r.label
}


features_cell = {
    'channel_cell_min': lambda r: np.min(masked(r,channel)),
    'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
    'channel_cell_mean' : lambda r: masked(r, channel).mean(),
    'channel_cell_median' : lambda r: np.median(masked(r, channel)),
    'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
    'channel_cell_max'    : lambda r: masked(r, channel).max(),
    'channel_cell_int'    : lambda r: masked(r, channel).sum(),
    'channel_cell_sd': lambda r: np.std(masked(r,channel)),
    'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
    'channel_zernike_cell': lambda r: mahotas_zernike(masked(r,channel)),
    'channel_pftas_cell': mahotas_pftas(masked(r,channel)),
    'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
    'dapi_channel_corr_cell': lambda r: correlate_channels(np.array([masked(r,i) for i in range(2)]),0,1),
    'area_cell'       : lambda r: r.area,
    'perimeter_cell' : lambda r: r.perimeter,
    'euler_cell' : lambda r: r.euler_number,
    'eccentricity_cell' : lambda r: r.eccentricity, #cell
    'major_axis_cell' : lambda r: r.major_axis_length, #cell
    'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
    'orientation_cell' : lambda r: r.orientation,
    'hu_moments_cell': lambda r: r.moments_hu,
    'solidity_cell': lambda r: r.solidity,
    'extent_cell': lambda r: r.extent,
    'cell'               : lambda r: r.label
    #Luke:add more dapi cell features?
}
