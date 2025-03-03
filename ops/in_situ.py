"""
Sequencing Data Processing and Base Calling Utilities

This module provides a set of functions for mapping and analyzing sequencing reads
(step 2 -- sbs base calling). It includes functions for:

1. Base Intensity Extraction: Functions to extract base intensities from sequencing data.
2. Base Calling: Algorithms for calling bases from raw intensity data.
3. Read Formatting: Tools to format and normalize sequencing reads.
4. Quality Score Calculation: Functions to compute quality scores for sequencing reads.
5. Barcode Assignment: Utilities for assigning barcodes to cells based on sequencing data.

"""

import numpy as np
import pandas as pd
from ops.constants import *
import ops.utils
import Levenshtein

def extract_base_intensity(maxed, peaks, cells, threshold_peaks):
    """
    Extract base intensity values, labels, and positions.

    Parameters:
        maxed (numpy.ndarray): Base intensity at each point, with dimensions (CYCLE, CHANNEL, I, J).
        peaks (numpy.ndarray): Peaks/local maxima score for each pixel.
        cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
        threshold_peaks (float): Threshold for identifying candidate sequencing reads based on peaks.

    Returns:
        tuple: Tuple containing values (base intensity values), labels (cell labels), and positions (positions of reads).
    """
    # Create a mask to identify reads outside of cells based on the peaks exceeding the threshold
    read_mask = (peaks > threshold_peaks)
    
    # Select base intensity values, corresponding labels, and positions for the identified reads
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T
    
    return values, labels, positions


def format_bases(values, labels, positions, cycles, bases):
    """
    Format extracted base intensity values, labels, and positions into a pandas DataFrame.

    Parameters:
        values (numpy.ndarray): Base intensity values extracted from the sequencing data.
        labels (numpy.ndarray): Labels corresponding to each read.
        positions (numpy.ndarray): Positions of the reads.
        cycles (int): Number of sequencing cycles.
        bases (list): List of bases corresponding to the sequencing channels.

    Returns:
        pandas.DataFrame: Formatted DataFrame containing base intensity values, labels, and positions.
    """
    index = (CYCLE, cycles), (CHANNEL, bases)
    
    try:
        # Attempt to reshape the extracted pixels to sequencing bases
        df = ops.utils.ndarray_to_dataframe(values, index)
    except ValueError:
        print('Failed to reshape extracted pixels to sequencing bases, writing empty table')
        return pd.DataFrame()

    # Create a DataFrame containing positions of the reads
    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
    
    # Stack the DataFrame to include cycles and bases as columns, reset the index, and rename columns
    df = (df.stack([CYCLE, CHANNEL])
          .reset_index()
          .rename(columns={0: INTENSITY, 'level_0': READ})
          .join(pd.Series(labels, name=CELL), on=READ)
          .join(df_positions, on=READ)
          .sort_values([CELL, READ, CYCLE])
          )

    return df


def do_median_call(df_bases, cycles=12, channels=4, correction_quartile=0, correction_only_in_cells=False, correction_by_cycle=False):
    """
    Call reads from raw base signal using median correction.

    Parameters:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.
        correction_quartile (int): Quartile used for correction.
        correction_only_in_cells (bool): Flag specifying whether correction is based on reads within cells or all reads.
        correction_by_cycle (bool): Flag specifying if correction should be done by cycle.

    Returns:
        pandas.DataFrame: DataFrame containing the called reads.
    """
    def correction(df, channels, correction_quartile, correction_only_in_cells):
        # Define the correction function
        if correction_only_in_cells:
            # Obtain transformation matrix W based on reads within cells
            X_ = dataframe_to_values(df.query('cell > 0'))
            _, W = transform_medians(X_.reshape(-1, channels), correction_quartile=correction_quartile)
            # Apply transformation to all data
            X = dataframe_to_values(df)
            Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
        else:
            # Apply correction to all data
            X = dataframe_to_values(df)
            Y, W = transform_medians(X.reshape(-1, channels), correction_quartile=correction_quartile)
        return Y, W

    # Apply correction either by cycle or to the entire dataset
    if correction_by_cycle:
        # Apply correction cycle by cycle
        Y = np.empty(df_bases.pipe(len), dtype=df_bases.dtypes['intensity']).reshape(-1, channels)
        for cycle, (_, df_cycle) in enumerate(df_bases.groupby('cycle')):
            Y[cycle::cycles, :], _ = correction(df_cycle, channels, correction_quartile, correction_only_in_cells)
    else:
        # Apply correction to the entire dataset
        Y, W = correction(df_bases, channels, correction_quartile, correction_only_in_cells)

    # Call barcodes
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads

def do_percentile_call(df_bases, cycles=12, channels=4, correction_only_in_cells=False, imaging_order='GTAC'):
    """
    Call reads from raw base signal using median correction. Use the
    `correction_within_cells` flag to specify if correction is based on reads within
    cells, or all reads.
    """
    #print(imaging_order)
    #print('nchannels ', channels)
    if correction_only_in_cells:
        # first obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query('cell > 0')) #seems to be superfluous arg, channels=channels)
        _, W = transform_percentiles(X_.reshape(-1, channels))

        # then apply to all data
        X = dataframe_to_values(df_bases) #superflous? , channels=channels)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases) #superflous? , channels=channels)
        Y, W = transform_percentiles(X.reshape(-1, channels))
        print(Y,Y.shape,X.shape,X)
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads

def transform_percentiles(X):
    """
    For each dimension, find points where that dimension is >=98th percentile intensity. Use median of those points to define new axes.
    Describe with linear transformation W so that W * X = Y.
    EDITED TO 95th percentile
    """

    def get_percentiles(X):
        arr = []
        for i in range(X.shape[1]):
            # print(X[:,i:i+1].shape)
            # print(X.shape)
            rowsums=np.sum(X,axis=1)[:,np.newaxis]
            X_rel = (X/rowsums)
            perc = np.nanpercentile(X_rel[:,i],95) #98)
            print(perc)
            high = X[X_rel[:,i] >= perc]
            print()
            #print(a.shape)
            #print((X/a)[0:2,:])
            #print((X[:,i:i+1]/(X+1e-6))[0:2,:])
            arr += [np.median(high, axis = 0)]
            #print(arr)
        M = np.array(arr)
        print(M)
        return M

    M = get_percentiles(X).T
    print(X)
    print(M)
    print(M.shape)
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W

def clean_up_bases(df_bases):
    """
    Sort DataFrame df_bases for pre-processing before dataframe_to_values.

    Parameters:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    # Sort DataFrame based on multiple columns
    return df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])


def call_cells(df_reads,
              df_UMI=None
              ):
    """
    Determine the count of top barcodes for each cell.

    Parameters:
        df_reads (pandas.DataFrame): DataFrame containing sequencing reads.

    Returns:
        pandas.DataFrame: DataFrame with the count of top barcodes for each cell.
    """
    cols = [WELL, TILE, CELL]
    s = (df_reads
         .drop_duplicates([WELL, TILE, READ])  # Drop duplicate reads
         .groupby(cols)[BARCODE]  # Group by well, tile, and cell, and barcode
         .value_counts()  # Count occurrences of each barcode within each group
         .rename('count')  # Rename the resulting series to 'count'
         .sort_values(ascending=False)  # Sort in descending order
         .reset_index()  # Reset the index
         .groupby(cols)  # Group again by well, tile, and cell
        )
    
    df_cells = (df_reads
          .join(s.nth(0)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': BARCODE_0}).set_index(cols), on=cols)
          .join(s.nth(0)[['well', 'tile', 'cell', 'count']].rename(columns={'count': BARCODE_COUNT_0}).set_index(cols), on=cols)
          .join(s.nth(1)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': BARCODE_1}).set_index(cols), on=cols)
          .join(s.nth(1)[['well', 'tile', 'cell', 'count']].rename(columns={'count': BARCODE_COUNT_1}).set_index(cols), on=cols)
          .join(s['count'].sum() .rename(BARCODE_COUNT),   on=cols)
          .assign(**{BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0).astype(int),
                     BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0).astype(int)})
          .drop_duplicates(cols)
          .drop([READ, BARCODE], axis=1) # drop the read
          .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
          .filter(regex='^(?!Q_)') # remove read quality scores
          .query('cell > 0') # remove reads not in a cell
        )

    if df_UMI is None:
        return df_cells
    else:
        return call_cells_add_UMIs(df_cells, df_UMI, cols=cols)
    
def call_cells_mapping(df_reads,
                       df_pool, 
                       error_correct=False,
                       barcode_info_cols = [SGRNA, GENE_SYMBOL, GENE_ID],
                       df_UMI=None,
                       **kwargs
                      ):
    """Determine the count of top barcodes, with prioritization given to barcodes mapping to the given pool design.
    
    Args:
        df_reads (DataFrame): DataFrame containing read data.
        df_pool (DataFrame): DataFrame containing pool design information.
        error_correct (Bool): Error correct barcodes against reference set. Default is False.
        barcode_info_cols (list, optional) : Columns related to barcode information
        df_UMI (DataFrame, optional): Dataframe containing UMI reads. Default is None (no UMIs)
    Returns:
        DataFrame: DataFrame containing the count of top barcodes along with merged guide information.
    """

    # optionally error correct barcodes
    # can be computationally expensive
    if error_correct:
        print('performing error correction')
        df_reads[BARCODE] = error_correct_reads(
            df_reads[BARCODE],
            df_pool[PREFIX],
            **kwargs,
        )
        
    # Map reads to the pool design
    df_mapped = (
        pd.merge(df_reads, df_pool[[PREFIX]], how='left', left_on=BARCODE, right_on=PREFIX)
        .assign(mapped=lambda x: pd.notnull(x[PREFIX]))  # Flag indicating if barcode is mapped
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )

    # Choose top 2 barcodes, priority given by (mapped, count)
    cols = [WELL, TILE, CELL]
    s = (
        df_mapped.drop_duplicates([WELL, TILE, READ])
        .groupby(cols + ['mapped'])[BARCODE]
        .value_counts()
        .rename('count')
        .reset_index()
        .sort_values(['mapped', 'count'], ascending=False)
        .groupby(cols)
    )

    # Create DataFrame containing top barcodes and their counts
    df_cells = (
        df_reads.join(s.nth(0)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': BARCODE_0}).set_index(cols), on=cols)
        .join(s.nth(0)[['well', 'tile', 'cell', 'count']].rename(columns={'count': BARCODE_COUNT_0}).set_index(cols), on=cols)
        .join(s.nth(1)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': BARCODE_1}).set_index(cols), on=cols)
        .join(s.nth(1)[['well', 'tile', 'cell', 'count']].rename(columns={'count': BARCODE_COUNT_1}).set_index(cols), on=cols)
        .join(s['count'].sum() .rename(BARCODE_COUNT),   on=cols)
        .assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0).astype(int),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)  # Remove duplicate rows
        .drop([READ, BARCODE], axis=1)  # Drop unnecessary columns
        .drop([POSITION_I, POSITION_J], axis=1)  # Drop the read coordinates
        .filter(regex='^(?!Q_)') # remove read quality scores
        .query('cell > 0')  # Remove reads not in a cell
    )

    # Merge guide information for barcode 0
    df_cells = (
        pd.merge(df_cells, df_pool[[PREFIX] + barcode_info_cols], how='left', left_on=BARCODE_0, right_on=PREFIX)
        .rename({col: col + '_0' for col in barcode_info_cols}, axis=1)  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )
    # Merge guide information for barcode 1
    df_cells = (
        pd.merge(df_cells, df_pool[[PREFIX] + barcode_info_cols], how='left', left_on=BARCODE_1, right_on=PREFIX)
        .rename({col: col + '_1' for col in barcode_info_cols}, axis=1)  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )
    
    if df_UMI is None:
        return df_cells
    else:
        return call_cells_add_UMIs(df_cells, df_UMI, cols=cols)


def call_cells_T7(
    df_reads, 
    df_pool=None, 
    map_col = 'prefix_map', 
    recomb_col = None,
    recomb_filter_col = None,
    recomb_q_thresh = 0.1,
    error_correct=False,
    barcode_info_cols = [GENE_SYMBOL],
    **kwargs
):
    """
    Determine the count of top barcodes for each cell, prioritizing the brightest spots per cell and ignoring spot counts. Intended for T7 IVT detection protocols.

    Parameters:
        df_reads (pandas.DataFrame): DataFrame containing sequencing reads.
        df_pool (None, pandas.DataFrame): DataFrame containing the reference barcode sequences. Default is None, in which case no mapping against a reference set is performed.
        map_col (str): The column within df_reads and df_pool containing the barcode for mapping
        recomb_col (None, str): Optional. The column within df_reads and df_pool containing the barcode for determining barcode recombination. Default is None, in which case no recombination identification is performed.
        recomb_filter_col (None, str): column to use for filtering recombination identification confidence. Default is None, in which case there is no quality filtering for recombination identification. e.g. Q_min of the subset of cycles used to determine recombination.
        recomb_q_thresh (float): Threshold for recomb_filter_col below which recombiation events will be indeterminate. Default is 0.5.
        error_correct (bool): Whether to perform error correction of barcodes.
        barcode_info_cols (list): list of columns from df_pool to maintain. e.g. "target" or "gene_symbol"
        
    Returns:
        pandas.DataFrame: DataFrame with the top barcodes for each cell.
    """
    
    # Choose top 2 barcodes, priority given by 'peak'
    cols = [WELL, TILE, CELL]

    # no mapping if df_pool is not provided
    if df_pool is None:
        s = df_reads.sort_values('peak', ascending=False).groupby(cols)

        df_cells = (
            df_reads.join(s.nth(0)[ cols+['barcode','peak'] ].rename(
                columns={'barcode':BARCODE_0,'peak':'peak_0'}).set_index(cols), on=cols)
            .join(s.nth(1)[ cols+['barcode','peak'] ].rename(
                columns={'barcode':BARCODE_1,'peak':'peak_1'}).set_index(cols), on=cols)
            .drop_duplicates(cols)
            .drop([READ, BARCODE, 'peak'], axis=1)
            .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
            .filter(regex='^(?!Q_)')
            .query('cell > 0')
        )
        return df_cells
    
    # optionally error correct barcodes
    # can be computationally expensive
    if error_correct:
        print('performing error correction')
        pre_correct_col = map_col+'_pre_correction'
        df_reads[pre_correct_col] = df_reads[map_col]
        df_reads[map_col] = error_correct_reads(
            df_reads[map_col],
            df_pool[map_col],
            **kwargs,
        )

    df_pool['map_temp'] = df_pool[map_col]
    df_mapped = (
        pd.merge(df_reads, df_pool[['map_temp']], how='left', left_on=map_col, right_on='map_temp')
        .assign(mapped=lambda x: pd.notnull(x['map_temp'])) # Flag indicating if barcode is mapped
        .drop('map_temp', axis=1)  # Drop the temporary prefix column
    )

    if recomb_col is not None:
        recomb_map = df_pool.set_index(map_col)[recomb_col].to_dict()
        df_mapped['no_recomb'] = (df_mapped[map_col].map(recomb_map)) == (df_mapped[recomb_col])
        # unmapped cells have undetermined recombination status
        df_mapped.loc[~df_mapped.mapped, 'no_recomb'] = np.nan
        df_mapped = df_mapped.drop(columns=[recomb_col])
        
        if recomb_filter_col is not None:
            # if below quality threshold, assigned undetermined recombination status
            df_mapped.loc[df_mapped[recomb_filter_col]<recomb_q_thresh, 'no_recomb'] = np.nan
    else:
        df_mapped['no_recomb']=np.nan
    
    s = (df_mapped
         .drop_duplicates([WELL, TILE, READ])
         .sort_values(['mapped','peak'], ascending=[True,False])
         .groupby(cols)
        )

    if error_correct:
        df_cells = (
            df_reads.join(s.nth(0)[cols+[map_col, pre_correct_col,'no_recomb','peak'] ].rename(
                columns={map_col:BARCODE_0, pre_correct_col:str('pre_correction_'+BARCODE_0), 'no_recomb':'no_recomb_0','peak':'peak_0'},
            ).set_index(cols), on=cols)
            .join(s.nth(1)[cols+[map_col,'no_recomb','peak'] ].rename(
                columns={map_col:BARCODE_1, pre_correct_col:str('pre_correction_'+BARCODE_1), 'no_recomb':'no_recomb_1','peak':'peak_1'}
            ).set_index(cols), on=cols)
            .drop_duplicates(cols)
            .drop([READ, BARCODE, map_col, pre_correct_col, recomb_col,'peak'], axis=1)
            .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
            .filter(regex='^(?!Q_)')
            .query('cell > 0')
        )

    else:
        df_cells = (
            df_reads.join(s.nth(0)[cols+[map_col,'no_recomb','peak'] ].rename(
                columns={map_col:BARCODE_0,'no_recomb':'no_recomb_0','peak':'peak_0'},
            ).set_index(cols), on=cols)
            .join(s.nth(1)[cols+[map_col,'no_recomb','peak'] ].rename(
                columns={map_col:BARCODE_1,'no_recomb':'no_recomb_1','peak':'peak_1'}
            ).set_index(cols), on=cols)
            .drop_duplicates(cols)
            .drop([READ, BARCODE, map_col, recomb_col,'peak'], axis=1)
            .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
            .filter(regex='^(?!Q_)')
            .query('cell > 0')
        )
    # for some reason this column is only sometimes found
    if 'no_recomb' in df_cells.columns:
        df_cells.drop(columns=['no_recomb'], inplace=True)
    
    # Merge guide information for barcode 0
    df_cells = (
        pd.merge(df_cells, df_pool[[map_col] + barcode_info_cols], how='left', left_on=BARCODE_0, right_on=map_col)
        .rename({col: col + '_0' for col in barcode_info_cols}, axis=1)  # Rename columns for clarity
        .drop(map_col, axis=1)  # Drop the temporary prefix column
    )
    # Merge guide information for barcode 1
    df_cells = (
        pd.merge(df_cells, df_pool[[map_col] + barcode_info_cols], how='left', left_on=BARCODE_1, right_on=map_col)
        .rename({col: col + '_1' for col in barcode_info_cols}, axis=1)  # Rename columns for clarity
        .drop(map_col, axis=1)  # Drop the temporary prefix column
    )

    return df_cells

def barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric='hamming'):
    """
    Expects list of barcodes for barcodes_1, optionally barcodes_2 as well.
    If barcodes_2 is False, find self-distances for barcodes_1.
    Returns distance matrix.
    """
    if distance_metric == 'hamming':
        distance = lambda i, j: Levenshtein.hamming(i, j)
    elif distance_metric == 'levenshtein':
        distance = lambda i, j: Levenshtein.distance(i, j)
    else:
        warnings.warn('distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"')
        distance = lambda i, j: Levenshtein.hamming(i, j)

    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    # create distance matrix for barcodes
    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix


def error_correct_reads(reads, reference, max_distance = 1, distance_metric = 'hamming'):
    """
    
    Error correct reads against a reference set of barcodes.
    Args:
        reads (pd.Series): Series with reads for error correction
        reference (pd.Series): Series with reference seqeuences
        max_distance (int, optional): The maximum distance for correction. Per-read, correction is performed only if (1) one 
        reference sequence is closest (no ties) and (2) that unique reference sequence is within the minimum distance. Default
        is 1.
        distance_metric (str, optional): Distance metric to compare barcodes. Options are 'hamming' (default) and 'levenshtein'.
    Returns:
        (pd.Series): corrected reads
    """

    # per-read distance to reference set
    dist_to_ref = barcode_distance_matrix(
        reads.to_list(),
        reference.to_list(),
        distance_metric = distance_metric,
    ) 
    
    # per read min distance to reference sequence 
    min_dist_to_ref = dist_to_ref.min(axis=1)

    # determine which corrections are unique
    unique_dist = np.array([ np.sum(dist_to_ref[x] == min_dist_to_ref[x])==1 for x in range(dist_to_ref.shape[0])])

    # filter the subset of reads that can be corrected
    corrected_subset = (unique_dist) & (min_dist_to_ref <= max_distance)
    
    # reads that can be corrected are corrected and returned
    # other reads are returned unmodified
    corrected_barcodes = reference.iloc[ dist_to_ref[corrected_subset].argmin(axis=1) ].values
    corrected_reads = reads.copy()
    corrected_reads.iloc[corrected_subset] = corrected_barcodes
    
    return corrected_reads

def call_cells_add_UMIs(df_cells, df_UMI, cols = [WELL, TILE, CELL]):
    """Determine the count of top barcodes, with prioritization given to barcodes mapping to the given pool design.
    
    Args:
        df_cells (DataFrame): DataFrame containing called cells.
        df_UMI (DataFrame): DataFrame containing UMI reads.
        cols (list, optional): List of columns to use to merge Dataframes
    Returns:
        DataFrame: df_cells DataFrame with top UMI counts.
    """
    s = (df_UMI
         .drop_duplicates([WELL, TILE, READ])  # Drop duplicate reads
         .groupby(cols)[BARCODE]  # Group by well, tile, and cell, and barcode
         .value_counts()  # Count occurrences of each barcode within each group
         .rename('count')  # Rename the resulting series to 'count'
         .sort_values(ascending=False)  # Sort in descending order
         .reset_index()  # Reset the index
         .groupby(cols)  # Group again by well, tile, and cell
        )
    
    df_cells_UMI = (df_UMI
      .join(s.nth(0)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': UMI_0}).set_index(cols), on=cols)
      .join(s.nth(0)[['well', 'tile', 'cell', 'count']].rename(columns={'count': UMI_COUNT_0}).set_index(cols), on=cols)
      .join(s.nth(1)[['well', 'tile', 'cell', 'barcode']].rename(columns={'barcode': UMI_1}).set_index(cols), on=cols)
      .join(s.nth(1)[['well', 'tile', 'cell', 'count']].rename(columns={'count': UMI_COUNT_1}).set_index(cols), on=cols)
      .join(s['count'].sum() .rename(UMI_COUNT),   on=cols)
      .assign(**{UMI_COUNT_0: lambda x: x[UMI_COUNT_0].fillna(0).astype(int),
                 UMI_COUNT_1: lambda x: x[UMI_COUNT_1].fillna(0).astype(int)})
      .drop_duplicates(cols)
      .drop([READ, BARCODE], axis=1) # drop the read
      .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
      .filter(regex='^(?!Q_)') # remove read quality scores
      .query('cell > 0') # remove reads not in a cell
    )
    cols_to_use = list(df_cells_UMI.columns.difference(df_cells.columns))

    return df_cells.merge(df_cells_UMI[cols_to_use + cols], left_on=cols, right_on=cols, how='inner')

def dataframe_to_values(df, value='intensity'):
    """
    Convert a sorted DataFrame containing intensity values into a 3D NumPy array.

    Parameters:
        df (pandas.DataFrame): DataFrame containing intensity values.
        value (str): Column name containing the intensity values.

    Returns:
        numpy.ndarray: 3D NumPy array representing intensity values with dimensions N x cycles x channels.
    """
    # Calculate the number of cycles
    cycles = df[CYCLE].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)

    # Calculate the number of channels
    n_channels = len(df[CHANNEL].value_counts())

    # Reshape intensity values into a 3D array
    x = np.array(df[value]).reshape(-1, n_cycles, n_channels)

    return x


def transform_medians(X, correction_quartile=0):
    """
    Compute a linear transformation matrix based on the median values of maximum points along each dimension of X.

    Parameters:
        X (numpy.ndarray): Input array.
        correction_quartile (float): Quartile used for correction.

    Returns:
        numpy.ndarray: Transformed array Y.
        numpy.ndarray: Transformation matrix W.
    """
    def get_medians(X, correction_quartile):
        arr = []
        for i in range(X.shape[1]):
            max_spots = X[X.argmax(axis=1) == i]
            try:
                arr.append(np.median(max_spots[max_spots[:, i] >= np.quantile(max_spots, axis=0, q=correction_quartile)[i]], axis=0))
            except:
                arr.append(np.median(max_spots, axis=0))
        M = np.array(arr)
        return M

    # Compute medians and construct matrix M
    M = get_medians(X, correction_quartile).T
    # Normalize matrix M
    M = M / M.sum(axis=0)
    # Compute the inverse of M to obtain the transformation matrix W
    W = np.linalg.inv(M)
    # Apply transformation to X
    Y = W.dot(X.T).T.astype(int)
    return Y, W

def normalize_bases(df):
    """
    Normalize the channel intensities by the median brightness of each channel in all spots.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing spot intensity data.

    Returns:
    --------
    pandas DataFrame
        DataFrame with normalized intensity values.
    """

    # Calculate median brightness of each channel
    df_medians = df.groupby('channel').intensity.median()

    # Normalize intensity values by dividing by respective channel median
    df_out = df.copy()
    df_out.intensity = df.apply(lambda x: x.intensity / df_medians[x.channel], axis=1)

    return df_out

def call_barcodes(df_bases, Y, cycles=12, channels=4):
    """
    Assign barcode sequences to reads based on the transformed base signal obtained from sequencing data.

    Parameters:
    - df_bases (pandas DataFrame): DataFrame containing base signal information for each read.
    - Y (numpy array): Transformed base signal reshaped into a suitable format for the calling process.
    - cycles (int): Number of sequencing cycles.
    - channels (int): Number of sequencing channels.

    Returns:
    - df_reads (pandas DataFrame): DataFrame with assigned barcode sequences and quality scores for each read.
    """
    # Extract unique bases
    bases = sorted(set(df_bases[CHANNEL]))
    
    # Check for weird bases
    if any(len(x) != 1 for x in bases):
        raise ValueError('supplied weird bases: {0}'.format(bases))
    
    # Remove duplicate entries and create a copy for storing barcode calls
    df_reads = df_bases.drop_duplicates([WELL, TILE, READ]).copy()
    
    # Call barcodes based on the transformed base signal
    df_reads[BARCODE] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)
    
    # Calculate quality scores for each read
    Q = quality(Y.reshape(-1, cycles, channels))
    
    # Store quality scores in DataFrame
    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:, i]
    
    # Assign minimum quality score for each read
    df_reads = df_reads.assign(Q_min=lambda x: x.filter(regex='Q_\d+').min(axis=1))
    
    # Drop unnecessary columns
    df_reads = df_reads.drop([CYCLE, CHANNEL, INTENSITY], axis=1)
    
    return df_reads

def call_bases_fast(values, bases):
    """
    Call bases based on the maximum intensity value for each cycle/channel combination.

    Parameters:
    - values (numpy array): 3D array containing intensity values for each cycle, channel, and base.
    - bases (str): String containing the base symbols corresponding to each channel.

    Returns:
    - calls (list of str): List of called bases for each read.
    """
    # Check dimensions and base length
    assert values.ndim == 3
    assert values.shape[2] == len(bases)
    
    # Determine the index of the maximum intensity value for each cycle/channel
    calls = values.argmax(axis=2)
    
    # Map the index to the corresponding base symbol
    calls = np.array(list(bases))[calls]
    
    # Combine the base symbols for each cycle to form the called bases for each read
    return [''.join(x) for x in calls]


def quality(X):
    """
    Calculate quality scores based on the intensity values.

    Parameters:
    - X (numpy array): Array containing intensity values.

    Returns:
    - Q (numpy array): Array containing quality scores.
    """
    # Sort the intensity values and convert to float
    X = np.abs(np.sort(X, axis=-1).astype(float))
    
    # Calculate the quality scores
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])
    
    # Clip the quality scores to the range [0, 1]
    Q = (Q * 2).clip(0, 1)
    
    return Q



def reads_to_fastq(df, microscope='MN', dataset='DS', flowcell='FC'):
    """
    Convert sequencing reads dataframe to FASTQ format.

    Parameters:
    - df (pandas DataFrame): DataFrame containing sequencing reads data.
    - microscope (str): Microscope identifier.
    - dataset (str): Dataset identifier.
    - flowcell (str): Flowcell identifier.

    Returns:
    - reads (list): List of reads in FASTQ format.
    """
    # Helper function to wrap fields with curly braces
    wrap = lambda x: '{' + x + '}'
    # Helper function to join fields with colon and wrap with curly braces
    join_fields = lambda xs: ':'.join(map(wrap, xs))

    # Construct FASTQ format template
    a = '@{m}:{d}:{f}'.format(m=microscope, d=dataset, f=flowcell)
    b = join_fields([WELL, CELL, 'well_tile', READ, POSITION_I, POSITION_J])
    c = '\n{b}\n+\n{{phred}}'.format(b=wrap(BARCODE))
    fmt = a + b + c 
    
    # Generate unique combinations of WELL and TILE
    well_tiles = sorted(set(df[WELL] + '_' + df[TILE]))
    fields = [WELL, TILE, CELL, READ, POSITION_I, POSITION_J, BARCODE]
    
    # Extract quality scores from the DataFrame
    Q = df.filter(like='Q_').values
    
    reads = []
    # Iterate over rows in the DataFrame and construct FASTQ reads
    for i, row in enumerate(df[fields].values):
        d = dict(zip(fields, row))
        d['phred'] = ''.join(phred(q) for q in Q[i])
        d['well_tile'] = well_tiles.index(d[WELL] + '_' + d[TILE])
        reads.append(fmt.format(**d))
    
    return reads


def dataframe_to_fastq(df, file, dataset):
    """
    Convert sequencing reads DataFrame to FASTQ format and write to a file.

    Parameters:
    - df (pandas DataFrame): DataFrame containing sequencing reads data.
    - file (str): Path to the output FASTQ file.
    - dataset (str): Dataset identifier.

    Returns:
    - None
    """
    # Convert DataFrame to FASTQ format
    s = '\n'.join(reads_to_fastq(df, dataset))
    # Write FASTQ format to file
    with open(file, 'w') as fh:
        fh.write(s)
        fh.write('\n')


def phred(q):
    """
    Convert quality score from 0...1 to ASCII Phred score.

    Parameters:
    - q (float): Quality score in the range 0 to 1.

    Returns:
    - str: ASCII Phred score.
    """
    n = int(q * 30 + 33)
    if n == 43:
        n += 1
    if n == 58:
        n += 1
    return chr(n)


def add_clusters(df_cells, barcode_col=BARCODE_0, radius=50,
                 verbose=True, ij=(POSITION_I, POSITION_J)):
    """
    Assign cluster labels to cells based on their spatial proximity.

    Parameters:
    - df_cells (pandas DataFrame): DataFrame containing cell information.
    - barcode_col (str): Column name containing barcode information.
    - radius (int): Radius within which cells are considered to be in the same cluster.
    - verbose (bool): Whether to print progress messages.
    - ij (tuple): Names of the columns containing spatial coordinates.

    Returns:
    - pandas DataFrame: DataFrame with additional columns for cluster labels and sizes.
    """
    from scipy.spatial.kdtree import KDTree
    import networkx as nx

    # Extract spatial coordinates and barcode information from DataFrame
    I, J = ij
    x = df_cells[GLOBAL_X] + df_cells[J]
    y = df_cells[GLOBAL_Y] + df_cells[I]
    barcodes = df_cells[barcode_col]
    barcodes = np.array(barcodes)

    # Construct KDTree for efficient spatial search
    kdt = KDTree(np.array([x, y]).T)
    num_cells = len(df_cells)

    # Print progress message if verbose mode is enabled
    if verbose:
        message = 'Searching for clusters among {} {} objects'
        print(message.format(num_cells, barcode_col))

    # Query pairs of cells within the specified radius
    pairs = kdt.query_pairs(radius)
    pairs = np.array(list(pairs))

    # Check if cells in each pair have the same barcode
    x = barcodes[pairs]
    y = x[:, 0] == x[:, 1]

    # Construct graph and find connected components to identify clusters
    G = nx.Graph()
    G.add_edges_from(pairs[y])
    clusters = list(nx.connected_components(G))

    # Assign cluster labels to cells
    cluster_index = np.zeros(num_cells, dtype=int) - 1
    for i, c in enumerate(clusters):
        cluster_index[list(c)] = i

    # Update DataFrame with cluster labels and sizes
    df_cells = df_cells.copy()
    df_cells[CLUSTER] = cluster_index
    df_cells[CLUSTER_SIZE] = (df_cells
        .groupby(CLUSTER)[barcode_col].transform('size'))
    df_cells.loc[df_cells[CLUSTER] == -1, CLUSTER_SIZE] = 1

    return df_cells


def index_singleton_clusters(clusters):
    """
    Index singleton clusters starting from the maximum cluster label.

    Parameters:
    - clusters (numpy array): Array containing cluster labels.

    Returns:
    - numpy array: Array with singleton clusters indexed from the maximum cluster label.
    """
    clusters = clusters.copy()

    # Identify singleton clusters and assign new labels starting from the maximum cluster label
    filt = clusters == -1
    n = clusters.max()
    clusters[filt] = range(n, n + len(filt))

    return clusters



def join_by_cell_location(df_cells, df_ph, max_distance=4):
    """
    Join dataframes based on cell location using a KDTree for efficient spatial querying.

    Parameters:
    - df_cells (pandas DataFrame): DataFrame containing cell data.
    - df_ph (pandas DataFrame): DataFrame containing photon data.
    - max_distance (int, optional): Maximum distance threshold for spatial join. Default is 4.

    Returns:
    - pandas DataFrame: Joined DataFrame containing cell and photon data.
    """
    from scipy.spatial.kdtree import KDTree

    # Extract coordinates for KDTree
    i_tree = df_ph['global_y']
    j_tree = df_ph['global_x']
    i_query = df_cells['global_y']
    j_query = df_cells['global_x']

    # Build KDTree for photon data
    kdt = KDTree(list(zip(i_tree, j_tree)))

    # Query KDTree with cell coordinates
    distance, index = kdt.query(list(zip(i_query, j_query)))

    # Retrieve corresponding cell IDs from photon DataFrame
    cell_ph = df_ph.iloc[index]['cell'].pipe(list)

    # Define columns for left and right DataFrames
    cols_left = ['well', 'tile', 'cell_ph']
    cols_right = ['well', 'tile', 'cell']
    cols_ph = [c for c in df_ph.columns if c not in df_cells.columns]

    # Join DataFrames based on cell location and distance threshold
    return (df_cells
            .assign(cell_ph=cell_ph, distance=distance)
            .query('distance < @max_distance')
            .join(df_ph.set_index(cols_right)[cols_ph], on=cols_left)
            )

