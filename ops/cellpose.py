"""Interface to cellpose
"""
from cellpose.models import Cellpose
import numpy as np
import contextlib
import sys

from ops.annotate import relabel_array
from skimage.measure import regionprops
from skimage.segmentation import clear_border



def segment_cellpose(dapi, cyto, nuclei_diameter, cell_diameter, gpu=False, 
                     net_avg=False, cyto_model='cyto', reconcile='consensus', logscale=True,
                     remove_edges=True):
    """How to disable the logger?
    """
    # import logging
    # logging.getLogger('cellpose').setLevel(logging.WARNING)

    if logscale:
        cyto = image_log_scale(cyto)
    img = np.array([dapi, cyto])

    model_dapi = Cellpose(model_type='nuclei', gpu=gpu, net_avg=net_avg)
    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu, net_avg=net_avg)
    
    nuclei, _, _, _ = model_dapi.eval(img, channels=[1, 0], diameter=nuclei_diameter)
    cells, _, _, _  = model_cyto.eval(img, channels=[2, 1], diameter=cell_diameter)

    if remove_edges:
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)

    print(f'found {nuclei.max()} nuclei before reconciling', file=sys.stderr)
    print(f'found {cells.max()} cells before reconciling', file=sys.stderr)
    if reconcile:
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
    print(f'found {cells.max()} nuclei/cells after reconciling', file=sys.stderr)

    return nuclei, cells


def segment_cellpose_rgb(rgb, nuclei_diameter, cell_diameter, gpu=False, 
                     net_avg=False, cyto_model='cyto', reconcile='consensus', logscale=True,
                     remove_edges=True):

    model_dapi = Cellpose(model_type='nuclei', gpu=gpu, net_avg=net_avg)
    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu, net_avg=net_avg)
    
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)
    cells, _, _, _  = model_cyto.eval(rgb, channels=[2, 3], diameter=cell_diameter)

    print(f'found {nuclei.max()} nuclei before removing edges', file=sys.stderr)
    print(f'found {cells.max()} cells before removing edges', file=sys.stderr)
    if remove_edges:
        print('removing edges')
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)

    print(f'found {nuclei.max()} nuclei before reconciling', file=sys.stderr)
    print(f'found {cells.max()} cells before reconciling', file=sys.stderr)
    if reconcile:
        print(f'reconciling masks with method how={reconcile}')
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
    print(f'found {cells.max()} nuclei/cells after reconciling', file=sys.stderr)

    return nuclei, cells

def segment_cellpose_nuclei_rgb(rgb, nuclei_diameter, gpu=False, 
                     net_avg=False, remove_edges=True, **kwargs):

    model_dapi = Cellpose(model_type='nuclei', gpu=gpu, net_avg=net_avg)
    
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)

    print(f'found {nuclei.max()} nuclei before removing edges', file=sys.stderr)
    if remove_edges:
        print('removing edges')
        nuclei = clear_border(nuclei)

    print(f'found {nuclei.max()} final nuclei', file=sys.stderr)

    return nuclei


def image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    import numpy as np
    data = data.astype(float)
    if ignore_zero:
        data_perc = data[data > 0]
    else:
        data_perc = data
    bottom = np.percentile(data_perc, bottom_percentile)
    data[data < bottom] = bottom
    scaled = np.log10(data - bottom + 1)
    # cut out the noisy bits
    floor = np.log10(floor_threshold)
    scaled[scaled < floor] = floor
    return scaled - floor


def reconcile_nuclei_cells(nuclei, cells, how='concensus' #erode_nuclei=5
    ):
    """Only keep nucleus, cell pairs that exclusively overlap each other. 
    Reindex both integer masks from 1.

    Parameters
    ----------

    nuclei : np.ndarray
        nuclei mask

    cells : np.ndarray
        cell mask

    how : str, {'concensus','contained_in_cells'}
        with 'consensus', cells are only kept if both cell and nuclei label matchings 
        are unique. With 'contained_in_cells', multiple nuclei for a single cell are kept 
        but merged.
    """
    from skimage.morphology import erosion

    def get_unique_label_map(regions,keep_multiple=False):
        d = {}
        for r in regions:
            masked = r.intensity_image[r.intensity_image > 0]
            labels = np.unique(masked)
            if keep_multiple:
                d[r.label] = labels
            elif len(labels) == 1:
                d[r.label] = labels[0]
        return d

    nuclei_eroded = center_pixels(nuclei)
    # erode, then set any changed pixel to background (separate touching regions)
    # selem = np.ones((erode_nuclei, erode_nuclei))
    # changed = erosion(nuclei, selem)
    # nuclei_eroded = nuclei.copy()
    # nuclei_eroded[nuclei != changed] = 0

    nucleus_map = get_unique_label_map(regionprops(nuclei_eroded, intensity_image=cells))
    if how=='contained_in_cells':
        cell_map = get_unique_label_map(regionprops(cells,  intensity_image=nuclei_eroded),keep_multiple=True)
    else:
        cell_map = get_unique_label_map(regionprops(cells,  intensity_image=nuclei_eroded))

    keep = []
    for nucleus in nucleus_map:
        try:
            if how=='contained_in_cells':
                if nucleus in cell_map[nucleus_map[nucleus]]:
                    keep += [[nucleus, nucleus_map[nucleus]]]
            else:
                if cell_map[nucleus_map[nucleus]] == nucleus:
                    keep += [[nucleus, nucleus_map[nucleus]]]
        except KeyError:
            pass

    if len(keep) == 0:
        return np.zeros_like(nuclei), np.zeros_like(cells)

    keep_nuclei, keep_cells = zip(*keep)

    if how=='contained_in_cells':
        nuclei = relabel_array(nuclei, {nuclei_label:cell_label for nuclei_label,cell_label in keep})
        cells[~np.isin(cells,keep_cells)] = 0
        labels,cell_indices = np.unique(cells,return_inverse=True)
        _,nuclei_indices = np.unique(nuclei,return_inverse=True)
        cells = np.arange(0,labels.shape[0])[cell_indices.reshape(*cells.shape)]
        nuclei = np.arange(0,labels.shape[0])[nuclei_indices.reshape(*nuclei.shape)]
    else:
        nuclei = relabel_array(nuclei, {label: i + 1 for i, label in enumerate(keep_nuclei)})
        cells  = relabel_array(cells,  {label: i + 1 for i, label in enumerate(keep_cells)})
    nuclei, cells = nuclei.astype(int), cells.astype(int)
    return nuclei, cells


def center_pixels(label_image):
    ultimate = np.zeros_like(label_image)
    for r in regionprops(label_image):
        i, j = np.array(r.bbox).reshape(2,2).mean(axis=0).astype(int)
        ultimate[i, j] = r.label
    return ultimate
