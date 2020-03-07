import inspect
import functools
import os
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')

import numpy as np
import pandas as pd
import skimage
import ops.features
import ops.process
import ops.io
import ops.in_situ
import ops.io_hdf
from ops.process import Align
from scipy.stats import mode
from ops.constants import *
from itertools import combinations


class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    @staticmethod
    def _calculate_illumination_correction(data, smooth=200, rescale=True):
        """calculate illumination correction field for use with apply_illumination_correction 
        Snake method. Equivalent to CellProfiler's CorrectIlluminationCalculate module with 
        option "Regular", "All", "Median Filter"
        Note: algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
        illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014)
        """
        data = np.array(data).astype(np.uint16) # dimensions of (FOV, CHANNEL, I, J)

        averaged = data.mean(axis=0).astype(np.uint16)

        selem = skimage.morphology.disk(smooth,dtype=np.uint16)

        median_filter = ops.utils.applyIJ(skimage.filters.median)

        smoothed = median_filter(averaged,selem)

        if rescale:
            # use 2nd percentile for robust minimum
            robust_mins = np.quantile(smoothed.reshape(smooth.shape[0],-1),q=0.02,axis=1)
            robust_mins[robust_mins==0] = 1

            smoothed = np.array([smoothed[ch]/robust_mins[ch] for ch in range(smoothed.shape[0])])
            smoothed[smoothed<1] = 1

        return smoothed

    @staticmethod
    def _apply_illumination_correction(raw, correction):
        corrected = (raw/correction).astype(np.uint16)

        return corrected

    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1,
        align_within_cycle=True, cycle_files=None,keep_trailing=False, n=1):
        """Rigid alignment of sequencing cycles and channels. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J). 'n' 
        determines the first SBS channel in 'data'.
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).
        """
        if isinstance(data,list):
            arr = []
            # snakemake passes de-nested list of numpy arrays
            current = 0
            for cycle in cycle_files:
                arr.append(np.array(data[current:current+cycle]))
                current += cycle
            # for element in data:
            #     if not isinstance(element,list):
            #         arr.append(element)
            #     elif len(element)==1:
            #         arr.append(element[0])
            #     else:
            #         arr.append(np.array(element))

            data = np.array(arr)
        else:
            data = np.array(data)

        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])

        if data.ndim==1:
            # data stacked with different cycle numbers?
            # assume extra channels exist are on the first cycle, first channels
            # does not return extra channels
            extra = data[0].shape[1] - data[1].shape[0]
            #stack channels in common
            stacked = np.concatenate([data[0][:,extra:],np.array([data[cycle] for cycle in range(1,data.shape[0])])],axis=0)
            #copy extra channels across other cycles
            stacked = np.concatenate((np.array([data[0][0,:extra]]*8),stacked),axis=1)
        else:
            extra = 0
            stacked = data

        assert stacked.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align between SBS channels for each cycle
        aligned = stacked.copy()
        if align_within_cycle:
            align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
            # if data.shape[1] == 4:
            #     n = 0
            #     align_it = lambda x: Align.align_within_cycle(x, window=window, 
            #         upsample_factor=upsample_factor, cutoff=cutoff)
            # else:
            #     n = 1
            
            aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])
            

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0, 
                                window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, n:], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(extra,aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2):
        """Align the second image to the first, using the channel at position 
        `channel_index`. If channel_index is a tuple of length 2, specifies channels of [data_1,data_2] 
        to use for alignment.The first channel is usually DAPI.
        """
        if isinstance(channel_index,tuple):
        	assert len(channel_index)==2, 'channel_index must either by an integer or tuple of length 2'
        	channel_index_1,channel_index_2 = channel_index
        else:
        	channel_index_1,channel_index_2 = (channel_index,)*2

        images = data_1[channel_index_1], data_2[channel_index_2]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        return aligned

    @staticmethod
    def _stack_channels(data):
        arr = []
        for dataset in data:
            if len(dataset.shape)>2:
                arr.extend([dataset[...,channel,:,:] for channel in range(dataset.shape[-3])])
            else:
                arr.append(dataset)
        return np.stack(arr,axis=-3)
        
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max,smooth=1.35,radius=15):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J).
        """
        if isinstance(data, list):
            dapi = data[0].astype(np.uint16)
        elif data.ndim == 3:
            dapi = data[0].astype(np.uint16)
        else:
            dapi = data.astype(np.uint16)

        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max, 
            smooth=smooth, radius=radius)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_stack(dapi, threshold, area_min, area_max):
        """Find nuclei from a nuclear stain (e.g., DAPI). Expects data to have shape (I, J) 
        (segments one image) or (N, I, J) (segments a series of DAPI images).
        """
        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        find_nuclei = ops.utils.applyIJ(ops.process.find_nuclei)
        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_cells(data, nuclei, threshold):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        """
        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].astype(np.uint16).min(axis=0).mean(axis=0)
        elif data.ndim == 3:
            mask = np.median(data[1:].astype(np.uint16), axis=0)
        elif data.ndim == 2:
            mask = data.astype(np.uint16)
        else:
            raise ValueError

        mask = mask > threshold
        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells

    @staticmethod
    def _segment_cells_tubulin(data, nuclei, threshold, area_min, area_max, radius=15,
        method='otsu', tubulin_channel=1, remove_boundary_cells=False, **kwargs):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        """
        if data.ndim == 3:
            tubulin = data[tubulin_channel].astype(np.uint16)
        elif data.ndim == 2:
            tubulin = data.astype(np.uint16)
        else:
            raise ValueError('input image has more than 3 dimensions')

        kwargs = dict(threshold=threshold, 
            area_min=area_min, 
            area_max=area_max, 
            radius=radius,
            method=method)

        kwargs.update(**kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = ops.process.find_tubulin_background(tubulin,nuclei,**kwargs)

        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = ops.process.find_cells(nuclei,mask,remove_boundary_cells=remove_boundary_cells)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        loged = ops.process.log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation to estimate sequencing read locations.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        leading_dims = tuple(range(0, data.ndim - 2))
        consensus = np.std(data, axis=leading_dims)

        return consensus
    
    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """Find local maxima and label by difference to next-highest neighboring
        pixel.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        if data.ndim == 2:
            data = [data]

        peaks = [ops.process.find_peaks(x, n=width) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """Apply a maximum filter in a window of `width`.
        """
        import scipy.ndimage.filters

        if data.ndim == 2:
            data = data[None, None]
        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (e.g., well and tile) and 
        label at that position in integer mask `cells`.
        """

        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_peaks))

        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        for k,v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases, correction_quartile=0, peaks=None, correction_only_in_cells=True, subtract_min=False):
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        """
        df_bases = df_bases.copy()
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return
        if subtract_min:
            df_bases['intensity'] = df_bases['intensity'] - df_bases.groupby([WELL,TILE,CELL,READ,CHANNEL])['intensity'].transform('min')
        
        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                correction_only_in_cells=correction_only_in_cells, correction_quartile=correction_quartile)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_cells(df_reads, df_pool=None,q_min=0):
        """Median correction performed independently for each tile.
        """
        if df_reads is None:
            return
        if df_pool is None:
            return (df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells))
        else:
            return (df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells_mapping,df_pool))

    @staticmethod
    def _extract_features(data, labels, wildcards, features=None,**kwargs):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        from ops.features import features_basic
        features = features.copy() if features else dict()
        features.update(features_basic)

        df = feature_table(data, labels, features)

        for k,v in sorted(wildcards.items()):
            df[k] = v
        
        return df

    @staticmethod
    def _extract_features_bare(data, labels, features=None,**kwargs):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        features = features.copy() if features else dict()
        features.update({'label': lambda r: r.label})

        df = feature_table(data, labels, features)

        return df

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)
             .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift_myc)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        if (nuclei.max() == 0) or (cells.max() == 0):
            return

        import ops.features

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n = (Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
            .rename(columns={'area': 'area_nuclear'}))

        df_c =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .drop(['i', 'j'], axis=1).rename(columns={'area': 'area_cell'}))


        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('label'), df_c.set_index('label')], axis=1, join='inner')
                .reset_index())

        return (df
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation_live(data, nuclei, wildcards):
        def _extract_phenotype_translocation_simple(data, nuclei, wildcards):
            import ops.features
            features = ops.features.features_translocation_nuclear_simple
            
            return (Snake._extract_features(data, nuclei, wildcards, features)
                .rename(columns={'label': 'cell'}))

        extract = _extract_phenotype_translocation_simple
        arr = []
        for i, (frame, nuclei_frame) in enumerate(zip(data, nuclei)):
            arr += [extract(frame, nuclei_frame, wildcards).assign(frame=i)]

        return pd.concat(arr)

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return (Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_geom(labels, wildcards):
        from ops.features import features_geom
        return Snake._extract_features(labels, labels, wildcards, features_geom)

    @staticmethod
    def _extract_simple_nuclear_morphology(data_phenotype, nuclei, wildcards):
        
        import ops.morphology_features

        df =  (Snake._extract_features(data_phenotype, nuclei, wildcards, ops.morphology_features.features_nuclear))

        return df

    @staticmethod
    def _extract_phenotype_cp(data_phenotype, nuclei, cells, wildcards, nucleus_channels='all', cell_channels='all', channel_names=['dapi','tubulin','gh2ax','phalloidin']):
        import ops.cp_emulator

        if nucleus_channels == 'all':
            try:
                nucleus_channels = list(range(data_phenotype.shape[-3]))
            except:
                nucleus_channels = [0]

        if cell_channels == 'all':
            try:
                cell_channels = list(range(data_phenotype.shape[-3]))
            except:
                cell_channels = [0]

        dfs = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_phenotype = data_phenotype.astype(np.uint16)

        # nucleus shape
        dfs.append(Snake._extract_features(nuclei,nuclei,wildcards,ops.cp_emulator.shape_features)
            .rename(columns=ops.cp_emulator.shape_columns)
            .set_index('label')
            .rename(columns = lambda x: 'nucleus_'+x if x not in wildcards.keys() else x)
            )

        # cell shape
        dfs.append(Snake._extract_features_bare(cells,cells,ops.cp_emulator.shape_features)
            .rename(columns=ops.cp_emulator.shape_columns)
            .set_index('label')
            .rename(columns = lambda x: 'cell_'+x)
            )

        # nucleus grayscale channel features
        dfs.extend([(Snake._extract_features_bare(data_phenotype[...,channel,:,:],nuclei,ops.cp_emulator.grayscale_features)
            .rename(columns=ops.cp_emulator.intensity_distribution_columns)
            .set_index('label')
            .rename(columns = lambda x: 'nucleus_'+channel_names[channel]+'_'+x)
            ) 
            for channel in nucleus_channels]
            )

        # cell grayscale channel features
        dfs.extend([(Snake._extract_features_bare(data_phenotype[...,channel,:,:],cells,ops.cp_emulator.grayscale_features)
            .rename(columns=ops.cp_emulator.intensity_distribution_columns)
            .set_index('label')
            .rename(columns = lambda x: 'cell_'+channel_names[channel]+'_'+x)
            ) 
            for channel in cell_channels]
            )

        # generate correlation column names

        ## nucleus
        nucleus_correlation_columns = {
        'colocalization_{}'.format(inner_num+outer_num*len(ops.cp_emulator.colocalization_columns))
        :col.format(first=channel_names[first],second=channel_names[second]) 
        for outer_num,(first,second) in enumerate(combinations(nucleus_channels,2)) 
        for inner_num,col in enumerate(ops.cp_emulator.colocalization_columns)
        }

        nucleus_correlation_columns.update({
        'correlation_{}'.format(num)
        :ops.cp_emulator.correlation_columns[0].format(first=channel_names[first],second=channel_names[second]) 
        for num,(first,second) in enumerate(combinations(nucleus_channels,2))
        })

        nucleus_correlation_columns.update({
        'lstsq_slope_{}'.format(num)
        :ops.cp_emulator.correlation_columns[1].format(first=channel_names[first],second=channel_names[second]) 
        for num,(first,second) in enumerate(combinations(nucleus_channels,2))
        })

        ## cell
        cell_correlation_columns = {
        'colocalization_{}'.format(inner_num+outer_num*len(ops.cp_emulator.colocalization_columns))
        :col.format(first=channel_names[first],second=channel_names[second]) 
        for outer_num,(first,second) in enumerate(combinations(cell_channels,2)) 
        for inner_num,col in enumerate(ops.cp_emulator.colocalization_columns)
        }

        cell_correlation_columns.update({
        'correlation_{}'.format(num)
        :ops.cp_emulator.correlation_columns[0].format(first=channel_names[first],second=channel_names[second]) 
        for num,(first,second) in enumerate(combinations(cell_channels,2))
        })

        cell_correlation_columns.update({
        'lstsq_slope_{}'.format(num)
        :ops.cp_emulator.correlation_columns[1].format(first=channel_names[first],second=channel_names[second]) 
        for num,(first,second) in enumerate(combinations(cell_channels,2))
        })


        # nucleus channel correlations
        dfs.append(Snake._extract_features_bare(data_phenotype[...,nucleus_channels,:,:],nuclei,ops.cp_emulator.correlation_features)
            .rename(columns=nucleus_correlation_columns)
            .set_index('label')
            .rename(columns = lambda x: 'nucleus_'+x)
            )

        # cell channel correlations
        dfs.append(Snake._extract_features_bare(data_phenotype[...,cell_channels,:,:],cells,ops.cp_emulator.correlation_features)
            .rename(columns=cell_correlation_columns)
            .set_index('label')
            .rename(columns = lambda x: 'cell_'+x)
            )

        # nucleus neighbors
        dfs.append(ops.cp_emulator.neighbor_measurements(nuclei,distances=[1,10])
            .set_index('label')
            .rename(columns = lambda x: 'nucleus_'+x)
            )

        # cell neighbors
        dfs.append(ops.cp_emulator.neighbor_measurements(cells,distances=[1,10])
            .set_index('label')
            .rename(columns = lambda x: 'cell_'+x)
            )

        return pd.concat(dfs,axis=1,join='outer',sort=True).reset_index()

    # @staticmethod
    # def _extract_phenotype_morphology(data_phenotype, nuclei, cells, wildcards):
        
    #     import ops.morphology_features
    #     # def masked(region, index):
    #     #     return region.intensity_image_full[index][region.filled_image]

    #     df_n =  (Snake._extract_features(data_phenotype, nuclei, wildcards, ops.morphology_features.features_nuclear)
    #              .drop(columns=list(wildcards.keys()))
    #             )

    #     df_n_to_c = (Snake._extract_features(cells,nuclei,wildcards,{'cell':lambda r: mode(r.intensity_image[r.intensity_image>0],axis=None).mode})
    #                  .rename({'label':'nucleus'},axis=1)
    #                  .drop(columns=(list(wildcards.keys())+['i','j','area']))
    #                 )

    #     df_n_full = df_n.merge(df_n_to_c,how='left',on='nucleus')

    #     df_c =  (Snake
    #              ._extract_features(data_phenotype, cells, wildcards, ops.morphology_features.features_cell)
    #              .drop(columns=['i','j','area','label'])
    #             ) 

    #     df = df_n_full.merge(df_c,how='left',on='cell')

    #     return df


    @staticmethod
    def _analyze_single(data, alignment_ref, cells, peaks, 
                        threshold_peaks, wildcards, channel_ix=1):
        if alignment_ref.ndim == 3:
            alignment_ref = alignment_ref[0]
        data = np.array([[alignment_ref, alignment_ref], 
                          data[[0, channel_ix]]])
        aligned = ops.process.Align.align_between_cycles(data, 0, window=2)
        loged = Snake._transform_log(aligned[1, 1])
        maxed = Snake._max_filter(loged, width=3)
        return (Snake._extract_bases(maxed, peaks, cells, bases=['-'],
                    threshold_peaks=threshold_peaks, wildcards=wildcards))

    @staticmethod
    def _track_live_nuclei(nuclei, tolerance_per_frame=5):
        
        # if there are no nuclei, we will have problems
        count = nuclei.max(axis=(-2, -1))
        if (count == 0).any():
            error = 'no nuclei detected in frames: {}'
            print(error.format(np.where(count == 0)))
            return np.zeros_like(nuclei)

        import ops.timelapse

        # nuclei coordinates
        arr = []
        for i, nuclei_frame in enumerate(nuclei):
            extract = Snake._extract_phenotype_minimal
            arr += [extract(nuclei_frame, nuclei_frame, {'frame': i})]
        df_nuclei = pd.concat(arr)

        # track nuclei
        motion_threshold = len(nuclei) * tolerance_per_frame
        G = (df_nuclei
          .rename(columns={'cell': 'label'})
          .pipe(ops.timelapse.initialize_graph)
        )

        cost, path = ops.timelapse.analyze_graph(G)
        relabel = ops.timelapse.filter_paths(cost, path, 
                                    threshold=motion_threshold)
        nuclei_tracked = ops.timelapse.relabel_nuclei(nuclei, relabel)

        return nuclei_tracked

    @staticmethod
    def _merge_triangle_hash(df_0,df_1,alignment):
        import ops.triangle_hash as th
        df_1 = df_1.rename(columns={'tile':'site'})
        model = th.build_linear_model(alignment['rotation'],alignment['translation'])
        return th.merge_sbs_phenotype(df_0,df_1,model)

    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], Snake.call_from_snakemake(f))

    @staticmethod
    def call_from_snakemake(f):
        """Turn a function that acts on a mix of image data, table data and other 
        arguments and may return image or table data into a function that acts on 
        filenames for image and table data, plus other arguments.

        If output filename is provided, saves return value of function.

        Supported input and output filetypes are .pkl, .csv, and .tif.
        """
        def g(**kwargs):

            # split keyword arguments into input (needed for function)
            # and output (needed to save result)
            input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)

            # load arguments provided as filenames
            input_kwargs = {k: load_arg(v) for k,v in input_kwargs.items()}

            results = f(**input_kwargs)

            if 'output' in output_kwargs:
                outputs = output_kwargs['output']
                
                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = '{0} output filenames provided for {1} results'
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

            else:
                return results 

        return functools.update_wrapper(g, f)


Snake.load_methods()


def remove_channels(data, remove_index):
    """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
    """
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    data = data[..., channels_mask, :, :]
    return data


# IO


def load_arg(x):
    """Try loading data from `x` if it is a filename or list of filenames.
    Otherwise just return `x`.
    """
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]
    nested_files = lambda x: [[load_file(f) for f in f_list] for f_list in x]
    
    for f in one_file, many_files, nested_files:
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # wasn't a file, probably a string arg
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # failed to load file
                return None
            pass
    else:
        return x


def save_output(filename, data, **kwargs):
    """Saves `data` to `filename`. Guesses the save function based on the
    file extension. Saving as .tif passes on kwargs (luts, ...) from input.
    """
    filename = str(filename)
    if data is None:
        # need to save dummy output to satisfy Snakemake
        with open(filename, 'w') as fh:
            pass
        return
    if filename.endswith('.tif'):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith('.pkl'):
        return save_pkl(filename, data)
    elif filename.endswith('.csv'):
        return save_csv(filename, data)
    elif filename.endswith('.hdf'):
        return save_hdf(filename, data)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def load_csv(filename):
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None
    return df


def load_pkl(filename):
    df = pd.read_pickle(filename)
    if len(df) == 0:
        return None


def load_tif(filename):
    return ops.io.read_stack(filename)

def load_hdf(filename):
    return ops.io_hdf.read_hdf_image(filename)

def save_csv(filename, df):
    df.to_csv(filename, index=None)


def save_pkl(filename, df):
    df.to_pickle(filename)


def save_tif(filename, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # `data` can be an argument name for both the Snake method and `save_stack`
    # overwrite with `data_` 
    kwargs['data'] = data_
    ops.io.save_stack(filename, **kwargs)

def save_hdf(filename,data_):
    ops.io_hdf.save_hdf_image(filename,data_)

def restrict_kwargs(kwargs, f):
    """Partition `kwargs` into two dictionaries based on overlap with default 
    arguments of function `f`.
    """
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard


def load_file(filename):
    """Attempt to load file, raising an error if the file is not found or 
    the file extension is not recognized.
    """
    if not isinstance(filename, str):
        raise TypeError
    if not os.path.isfile(filename):
        raise IOError(2, 'Not a file: {0}'.format(filename))
    if filename.endswith('.tif'):
        return load_tif(filename)
    elif filename.endswith('.pkl'):
        return load_pkl(filename)
    elif filename.endswith('.csv'):
        return load_csv(filename)
    elif filename.endswith('.hdf'):
        return load_hdf(filename)
    else:
        raise IOError(filename)


def get_arg_names(f):
    """List of regular and keyword argument names from function definition.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    """Get the kwarg defaults as a dictionary.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def load_well_tile_list(filename):
    if filename.endswith('pkl'):
        wells, tiles = pd.read_pickle(filename)[['well', 'tile']].values.T
    elif filename.endswith('csv'):
        wells, tiles = pd.read_csv(filename)[['well', 'tile']].values.T
    return wells, tiles
