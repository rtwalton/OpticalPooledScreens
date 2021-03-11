import numpy as np
import ops.utils

# FUNCTIONS


def correlate_channels(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A, B = r.intensity_image_full[[first, second]]

    filt = A > 0
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()

# use with ops.utils.regionprops_multichannel
def correlate_channels_multichannel(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A, B = r.intensity_image[...,[first, second]].T

    filt = (A > 0)&(B > 0)
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()

# use with ops.utils.regionprops_multichannel
def correlate_channels_all_multichannel(r):
    """Cross-correlation between masked images of all channels.
    """
    R = np.corrcoef(r.intensity_image[r.image].T)
    # same order as itertools.combinations of channel numbers
    return R[np.triu_indices_from(R,k=1)]

def masked(r, index):
    return r.intensity_image_full[index][r.image]

# use with ops.utils.regionprops_multichannel
def masked_multichannel(r, index):
    return r.intensity_image[r.image,index]

def correlate_channels_masked(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A = masked(r,first)
    B = masked(r, second)

    filt = (A > 0)&(B > 0)
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()

def count_labels(labels,return_list=False):
    uniques = np.unique(labels)
    ls = np.delete(uniques,np.where(uniques==0))
    if return_list:
        return len(ls),ls
    return len(ls)

# FEATURES
# these functions expect an `skimage.measure.regionprops` region as input

intensity = {
    'mean': lambda r: r.intensity_image[r.image].mean(),
    'median': lambda r: np.median(r.intensity_image[r.image]),
    'max': lambda r: r.intensity_image[r.image].max(),
    'min': lambda r: r.intensity_image[r.image].min(),
    }

# use with ops.utils.regionprops_multichannel
intensity_multichannel = {
    'mean': lambda r: r.intensity_image[r.image].mean(axis=0),
    'median': lambda r: np.median(r.intensity_image[r.image],axis=0),
    'max': lambda r: r.intensity_image[r.image].max(axis=0),
    'min': lambda r: r.intensity_image[r.image].min(axis=0),
    }

geometry = {
    'area'    : lambda r: r.area,
    'i'       : lambda r: r.centroid[0],
    'j'       : lambda r: r.centroid[1],
    'bounds'  : lambda r: r.bbox,
    # 'contour' : lambda r: ops.utils.binary_contours(r.image, fix=True, labeled=False)[0],
    'label'   : lambda r: r.label,
    # 'mask':     lambda r: ops.utils.Mask(r.image),
    'eccentricity': lambda r: r.eccentricity,
    'solidity': lambda r: r.solidity,
    'convex_area': lambda r: r.convex_area,
    'perimeter': lambda r: r.perimeter
    }

# DAPI, HA, myc
frameshift = {
    'dapi_ha_corr' : lambda r: correlate_channels_masked(r, 0, 1),
    'dapi_myc_corr': lambda r: correlate_channels_masked(r, 0, 2),
    'ha_median'    : lambda r: np.median(masked(r,1)),
    'myc_median'   : lambda r: np.median(masked(r,2)),
    'cell'         : lambda r: r.label,
    }

# use with ops.utils.regionprops_multichannel
frameshift_multichannel = {
    'dapi_ha_corr' : lambda r: correlate_channels_multichannel(r, 0, 1),
    'dapi_myc_corr': lambda r: correlate_channels_multichannel(r, 0, 2),
    'ha_myc_medians'    : lambda r: np.median(r.intensity_image[r.image,1:3],axis=0),
    'cell'         : lambda r: r.label,
    }

translocation = {
    'dapi_gfp_corr' : lambda r: correlate_channels_masked(r, 0, 1),
    'dapi_mean'  : lambda r: masked(r, 0).mean(),
    'dapi_median': lambda r: np.median(masked(r, 0)),
    'gfp_median' : lambda r: np.median(masked(r, 1)),
    'gfp_mean'   : lambda r: masked(r, 1).mean(),
    'dapi_int'   : lambda r: masked(r, 0).sum(),
    'gfp_int'    : lambda r: masked(r, 1).sum(),
    'dapi_max'   : lambda r: masked(r, 0).max(),
    'gfp_max'    : lambda r: masked(r, 1).max(),
    }

# use with ops.utils.regionprops_multichannel
translocation_multichannel = {
    'dapi_gfp_corr' : lambda r: correlate_channels_multichannel(r, 0, 1),
    'dapi_gfp_means' : lambda r: r.intensity_image[r.image,:2].mean(axis=0),
    'dapi_gfp_medians': lambda r: np.median(r.intensity_image[r.image,:2],axis=0),
    'dapi_gfp_ints' : lambda r: r.intensity_image[r.image,:2].sum(axis=0),
    'dapi_gfp_maxs' : lambda r: r.intensity_image[r.image,:2].max(axis=0)
    }

foci = {
    'foci_count' : lambda r: count_labels(r.intensity_image),
    'foci_area' : lambda r: (r.intensity_image>0).sum(),
    }

viewRNA = {
    'cy3_median': lambda r: np.median(masked(r, 1)),
    'cy5_median': lambda r: np.median(masked(r, 2)),
    'cy5_80p'   : lambda r: np.percentile(masked(r, 2), 80),
    'cy3_int': lambda r: masked(r, 1).sum(),
    'cy5_int': lambda r: masked(r, 2).sum(),
    'cy5_mean': lambda r: masked(r, 2).sum(),
    'cy5_max': lambda r: masked(r, 2).max(),
}

all_features = [
    intensity, 
    geometry,
    translocation,
    frameshift,
    viewRNA
    ]


def validate_features():
    names = sum(map(list, all_features), [])
    assert len(names) == len(set(names))

def make_feature_dict(feature_names):
    features = {}
    [features.update(d) for d in all_features]
    return {n: features[n] for n in feature_names}

validate_features()

features_basic = make_feature_dict(('area', 'i', 'j', 'label','bounds'))

features_geom = make_feature_dict((
    'area', 'eccentricity', 'convex_area', 'perimeter'))

features_translocation_nuclear = make_feature_dict((
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_translocation_cell = make_feature_dict((	
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_frameshift = make_feature_dict((
    'dapi_ha_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median'))

features_frameshift_myc = make_feature_dict((
    'dapi_ha_corr', 'dapi_myc_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median', 'myc_median'))

features_translocation_nuclear_simple = make_feature_dict((
	'dapi_gfp_corr', 
	'dapi_mean', 'dapi_max', 'gfp_mean', 'gfp_max',
    'area'))