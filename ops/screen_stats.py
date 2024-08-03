"""
Screen Statistics and Data Analysis Utilities

This module provides a collection of functions for analyzing understanding and visualizing features
extracted from pooled screening data (relating to 4 -- aggregate). It includes functions for:

1. Data aggregation: Computing statistics and metrics for cell populations across different conditions.
2. Statistical analysis: Performing statistical tests and calculating enrichment scores.
3. Visualization: Plotting distribution functions and feature correlation networks.
4. Feature processing: Applying transformations and testing for normality.
5. Correlation analysis: Identifying and visualizing correlated features and their network structure.

"""

import numpy as np
import pandas as pd
from random import choice,choices
from itertools import combinations

from ops.constants import *
from ops.utils import groupby_histogram, groupby_reduce_concat
from scipy.stats import wasserstein_distance, ks_2samp, ttest_ind, kstest

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdm_auto
from joblib import Parallel,delayed

## AGGREGATION

def distribution_difference(df, col='dapi_gfp_corr', control_query='gene_symbol == "non-targeting"', groups='gene_symbol'):
    """
    Calculate the Wasserstein distance between each group and a control group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name for the values to compare. Default is 'dapi_gfp_corr'.
        control_query (str): Query string to select the control group. Default is 'gene_symbol == "non-targeting"'.
        groups (str or list): Column(s) to group by. Default is 'gene_symbol'.

    Returns:
        pd.Series: Wasserstein distances for each group compared to the control.
    """
    y_neg = df.query(control_query)[col]
    return df.groupby(groups).apply(lambda x: wasserstein_distance(x[col], y_neg))

def process_rep(df, value='dapi_gfp_corr_nuclear', 
                sgRNA_index=('sgRNA_name', 'gene_symbol'),
                control_query='gene_symbol=="nontargeting"'):
    """
    Calculate statistics for one replicate.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value (str): Column name for the values to analyze. Default is 'dapi_gfp_corr_nuclear'.
        sgRNA_index (tuple): Columns to group by for sgRNA. Default is ('sgRNA_name', 'gene_symbol').
        control_query (str): Query string to select the control group. Default is 'gene_symbol=="nontargeting"'.

    Returns:
        pd.DataFrame: DataFrame with calculated statistics.

    Example:
        sample_index = ['replicate', 'stimulant', 'well']
        genes = ['MYD88', 'TRADD', 'nontargeting']
        stats = (df_cells
         .groupby(sample_index)
         .apply(process_rep).reset_index()
        )
    """
    sgRNA_index = list(sgRNA_index)
    nt = df.query(control_query)[value]
    w_dist = lambda x: wasserstein_distance(x, nt)
    ks_test = lambda x: ks_2samp(x, nt)
    t_test = lambda x: ttest_ind(x, nt)
    return (df
     .groupby(sgRNA_index)[value]
     .pipe(groupby_reduce_concat, 'mean', 'count', 
           w_dist=w_dist, ks_test=ks_test, t_test=t_test)
     .assign(ks_pval=lambda x: x['ks_test'].apply(lambda y: y.pvalue))
     .assign(ks_stat=lambda x: x['ks_test'].apply(lambda y: y.statistic))
     .assign(ttest_pval=lambda x: x['t_test'].apply(lambda y: y.pvalue))
     .assign(ttest_stat=lambda x: x['t_test'].apply(lambda y: y.statistic))
     .drop(columns=['ks_test','t_test'])
    )

def get_simple_stats(df_stats):
    """
    Calculate simple statistics from a DataFrame of statistics.

    Args:
        df_stats (pd.DataFrame): Input DataFrame with statistics.

    Returns:
        pd.DataFrame: DataFrame with calculated means and ranks for different stimulants.
    """
    return (df_stats
     .groupby(['gene_symbol', 'stimulant'])
     .apply(lambda x: x.eval('mean * count').sum() / x['count'].sum())
     .rename('mean')
     .reset_index()
     .pivot_table(index='gene_symbol', columns='stimulant', values='mean')
     .assign(IL1b_rank=lambda x: x['IL1b'].rank().astype(int))
     .assign(TNFa_rank=lambda x: x['TNFa'].rank().astype(int))
    )

def enrichment_score(sorted_metric, ranks, p=1, return_running=False):
    """
    Calculate the enrichment score for a sorted metric and given ranks.

    Args:
        sorted_metric (np.array): Pre-sorted metric values.
        ranks (np.array): Ranks of interest, computed with method='first'.
        p (int): Power for the enrichment score calculation. Default is 1.
        return_running (bool): If True, return running sum and leading edge. Default is False.

    Returns:
        float or tuple: Enrichment score, or tuple of running sum and leading edge if return_running is True.
    """
    P = np.ones(len(sorted_metric)) * (-1 / (len(sorted_metric) - len(ranks)))
    
    N_R = (abs(sorted_metric[ranks])**p).sum()
    P[ranks] = abs(sorted_metric[ranks])**p / N_R
    
    P_running = P.cumsum()
    
    leading_edge = np.argmax(abs(P_running))
    
    if return_running:
        return P_running, leading_edge
    return P_running[leading_edge]

def enrichment_score_minimal(selected_metric, selected_ranks, N, p=1):
    """
    Calculate enrichment score using minimal information.

    Args:
        selected_metric (np.array): Metric values for the selected set.
        selected_ranks (np.array): Ranks for the selected set.
        N (int): Total number of genes.
        p (int): Power for the enrichment score calculation. Default is 1.

    Returns:
        float: Enrichment score.
    """
    P = np.ones(N) * (-1 / (N - len(selected_ranks)))
    
    N_R = (abs(selected_metric)**p).sum()
    P[sorted(selected_ranks)] = (np.abs(sorted(selected_metric, reverse=True))**p) / N_R
    
    P_running = P.cumsum()
    
    leading_edge = np.argmax(abs(P_running))
    
    return P_running[leading_edge]

def aggregate_enrichment_score(df, grouping, cols=None, p=1):
    """
    Calculate aggregate enrichment scores for multiple columns and groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        grouping (str or list): Column(s) to group by.
        cols (list): Columns to calculate enrichment scores for. If None, use all columns.
        p (int): Power for the enrichment score calculation. Default is 1.

    Returns:
        pd.DataFrame: DataFrame with aggregated enrichment scores.
    """
    if cols is None:
        cols = df.columns
    df = pd.concat([df, df[cols].rank(ascending=False, method='first').astype(int).add_suffix('_rank') - 1], axis=1)
    
    N = df.pipe(len)
    
    arr = []
    for col in tqdm(cols):
        arr.append(df
                   .groupby(grouping)
                   [[col, f'{col}_rank']]
                   .apply(lambda x: enrichment_score_minimal(x.iloc[:,0].values, x.iloc[:,1].values, N, p=p))
                   .rename(col)
                  )
    return pd.concat(arr, axis=1)

## PLOTTING

def plot_distributions(df_cells, gene, col='dapi_gfp_corr_nuclear',
    control_query='gene_symbol=="nt"', replicate_col='replicate', conditions_col='stimulant',
    conditions = ['TNFa', 'IL1b'], range=(-1,1), n_bins=100
    ):
    """
    Plot cumulative distribution functions for a specific gene and control across replicates and conditions.

    Args:
        df_cells (pd.DataFrame): DataFrame containing cell data.
        gene (str): Gene symbol to plot.
        col (str): Column name for the values to plot. Default is 'dapi_gfp_corr_nuclear'.
        control_query (str): Query string to select control cells. Default is 'gene_symbol=="nt"'.
        replicate_col (str): Column name for replicates. Default is 'replicate'.
        conditions_col (str): Column name for conditions. Default is 'stimulant'.
        conditions (list): List of conditions to plot. Default is ['TNFa', 'IL1b'].
        range (tuple or str): Range for histogram bins. If 'infer', use data range. Default is (-1,1).
        n_bins (int): Number of bins for histogram. Default is 100.

    Returns:
        sns.FacetGrid: Seaborn FacetGrid object with the plotted distributions.
    """
    df_neg = (df_cells
     .query(control_query).assign(sgRNA='nontargeting'))
    df_gene = df_cells.query('gene_symbol == @gene')
    df_plot = pd.concat([df_neg, df_gene])
    replicates = sorted(set(df_plot[replicate_col]))
    if range=='infer':
        range = (df_plot[col].min(),df_plot[col].max())
    bins = np.linspace(range[0], range[1], n_bins)
    hist_kws = dict(bins=bins, 
        histtype='step', density=True, 
                    cumulative=True)
    row_order = conditions
    fg = (df_plot
     .pipe(sns.FacetGrid, hue='sgRNA', col_order=replicates,
           col=replicate_col, row=conditions_col, row_order=conditions)
     .map(plt.hist, col, **hist_kws)
    )
    
    return fg

def plot_distributions_nfkb(df_cells, gene):
    """
    Plot cumulative distribution functions for NF-kB correlation in a specific gene and control across replicates and stimulants.

    This function is a specific implementation of plot_distributions with preset parameters for NF-kB analysis.

    Args:
        df_cells (pd.DataFrame): DataFrame containing cell data.
        gene (str): Gene symbol to plot.

    Returns:
        sns.FacetGrid: Seaborn FacetGrid object with the plotted distributions.
    """
    df_neg = (df_cells
     .query('gene_symbol == "nt"').assign(sgRNA_name='nt'))
    df_gene = df_cells.query('gene_symbol == @gene')
    df_plot = pd.concat([df_neg, df_gene])
    replicates = sorted(set(df_plot['replicate']))
    bins = np.linspace(-1, 1, 100)
    hist_kws = dict(bins=bins, histtype='step', density=True, 
                    cumulative=True)
    row_order = 'TNFa', 'IL1b'
    fg = (df_plot
     .pipe(sns.FacetGrid, hue='sgRNA_name', col_order=replicates,
           col='replicate', row='stimulant', row_order=row_order)
     .map(plt.hist, 'dapi_gfp_corr_nuclear', **hist_kws)
    )
    
    return fg

## FEATURE PROCESSING

def generalized_log(y, offset=0):
    """
    Apply a generalized logarithm transformation to the input data.

    Args:
        y (np.array): Input data to transform.
        offset (float): Offset parameter for the transformation. Default is 0.

    Returns:
        np.array: Transformed data.
    """
    return np.log((y + np.sqrt(y**2 + offset**2))/2)

def feature_normality_test(df, columns='all'):
    """
    Test for normality of feature distributions using the Kolmogorov-Smirnov test.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        columns (list or str): Columns to test. If 'all', test all columns. Default is 'all'.

    Returns:
        pd.DataFrame: DataFrame with normality test results for each feature.
    """
    if columns == 'all':
        columns = df.columns
        
    results = []
    
    for col in columns:
        values = df[col].values
        standardized = (values-values.mean())/values.std()
        ks_result = kstest(standardized, 'norm')
        results.append({'feature': col, 'ks_statistic': ks_result[0], 'p_value': ks_result[1]})
        
    return pd.DataFrame(results)

def get_feature_pair_correlations(df_corr):
    """
    Get pairwise correlations between features from a correlation matrix.

    Args:
        df_corr (pd.DataFrame): Correlation matrix of features.

    Returns:
        pd.DataFrame: DataFrame with pairwise correlations and feature names.
    """
    return (pd.DataFrame(
        df_corr.values[np.triu(np.ones(df_corr.shape,dtype=bool),k=1)], columns=['correlation'])
        .assign(**{f'feature_{n}': features 
            for n, features in enumerate(zip(*tuple(combinations(df_corr.columns,2))))})
        )

def get_feature_correlation_connected_components(df_corr, threshold=0.9):
    """
    Find connected components in the feature correlation graph.

    Args:
        df_corr (pd.DataFrame): Correlation matrix of features.
        threshold (float): Correlation threshold for connecting features. Default is 0.9.

    Returns:
        pd.DataFrame: DataFrame with features and their corresponding connected component.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    graph = csr_matrix(((df_corr.abs()>threshold).values-np.eye(df_corr.pipe(len))))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return pd.DataFrame({'feature': df_corr.columns, 'component': labels})

def get_feature_pair_info(df, threshold=0.9):
    """
    Get information about feature pairs, including correlations and connected components.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        threshold (float): Correlation threshold for connecting features. Default is 0.9.

    Returns:
        pd.DataFrame: DataFrame with feature pair information.
    """
    df_corr = df.corr()
    df_pairs = get_feature_pair_correlations(df_corr)
    df_components = get_feature_correlation_connected_components(df_corr, threshold=threshold)
    return (df_pairs
        .merge(df_components.add_suffix('_0'), how='left', on='feature_0')
        .merge(df_components.add_suffix('_1'), how='left', on='feature_1')
        )

def visualize_connected_components(df, threshold=0.9, scale=0.5):
    """
    Visualize connected components of the feature correlation graph.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        threshold (float): Correlation threshold for connecting features. Default is 0.9.
        scale (float): Scale parameter for the spring layout. Default is 0.5.

    Returns:
        tuple: (G, positions, fig, ax) where G is the NetworkX graph, positions are the node positions,
               fig is the matplotlib figure, and ax are the axes of the plot.
    """
    import networkx as nx 
    df_corr = df.corr().abs()
    df_corr[df_corr<threshold] = 0
    G = nx.from_numpy_matrix((df_corr.values-np.eye(len(df.columns))))
    positions = nx.spring_layout(G, scale=scale)
    position_values = np.array(list(positions.values()))
    x_max, y_max = position_values.max(axis=0)
    x_min, y_min = position_values.min(axis=0)
    x_margin = (x_max-x_min)*0.25
    y_margin = (y_max-y_min)*0.25
    labels = '\n'.join([f'{node}: {label}' for label,node in zip(df.columns,positions.keys())])
    label_numbers  = {node:node for node in positions.keys()}
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    nx.draw_networkx(G, pos=positions, labels=label_numbers, node_size=50, ax=ax[0])
    ax[0].set_xlim(x_min-x_margin, x_max+x_margin)
    ax[0].set_ylim(y_min-y_margin, y_max+y_margin)
    ax[0].axis('on')
    ax[1].text(0, 0.5, labels, horizontalalignment='left', verticalalignment='center', transform=ax[1].transAxes)
    ax[1].axis('off')
    return G, positions, fig, ax