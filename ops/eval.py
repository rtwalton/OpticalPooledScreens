"""
Cluster Evaluation and Gene Pair Analysis Utilities

This module provides functions for evaluating gene clusters and analyzing gene pairs,
particularly in the context of comparing clustering results with known annotations
(relating to step 5 -- clustering). It includes functions for:

1. Precision and Recall Calculation: Functions to compute precision and recall metrics for gene clusters.
2. Gene Pair Expansion: Tools to handle gene symbols containing ampersands and expand them into multiple entries.
3. Distance Matrix Processing: Utilities to convert distance matrices into pairwise distance DataFrames.
4. Annotation-based Metrics: Functions to calculate recall and precision based on gene annotations.
5. Cluster Comparison: Tools to compare clustering results with known gene complex or pathway annotations.

"""

import itertools
import numpy as np
import pandas as pd

def precision_recall_cluster_pairs(df, annotation_pairs_match, annotation_singles_match, annotation_name='complex'):
    """
    Calculate precision and recall metrics for gene clusters.
    
    Parameters:
        df (pd.DataFrame): DataFrame with cluster information.
        annotation_pairs_match (pd.DataFrame): Annotations for gene pairs.
        annotation_singles_match (pd.DataFrame): Annotations for individual genes.
        annotation_name (str): Column name for the annotation.

    Returns:
        dict: Dictionary with precision and recall metrics.
    """
    # Basic cluster statistics
    results = {
        'n_clusters': df['cluster'].nunique(),
        'singularities': (df['cluster'].value_counts() == 1).sum(),
        'mean_cluster_size': df['cluster'].value_counts().mean()
    }

    arr_ = []

    # Generate pairs of genes within each cluster
    for c, df_cluster in df.reset_index().groupby(['cluster']):
        for (one, two) in itertools.combinations(df_cluster.to_dict('records'), 2):
            arr_.append({
                'gene_id_A': one['gene_id'], 'gene_symbol_A': one['gene_symbol'],
                'gene_id_B': two['gene_id'], 'gene_symbol_B': two['gene_symbol'],
                'cluster': c
            })

    df_cluster_pairs = pd.DataFrame(arr_)
    
    try:
        # Expand gene symbols with ampersands
        df = expand_ampersand_singles(df.reset_index())
        df_cluster_pairs = expand_ampersand_pairs(df_cluster_pairs)
    except:
        pass
    
    print('Mapping to annotations')

    # Merge with annotation data
    df_cluster_pairs = (
        df_cluster_pairs
        .astype({'gene_id_A': int, 'gene_id_B': int})
        .merge(
            annotation_singles_match.drop(columns=['gene_symbol']).add_suffix('_A'),
            how='left', on=['gene_id_A'], validate='many_to_many'
        )
        .merge(
            annotation_singles_match.drop(columns=['gene_symbol']).add_suffix('_B'),
            how='left', on=['gene_id_B'], validate='many_to_many'
        )
    )

    df_annotation_pairs_cluster_ = (
        annotation_pairs_match
        .rename(columns={'GeneA': 'gene_id_A', 'GeneB': 'gene_id_B'})
        .merge(
            df[['gene_id', 'cluster']].astype({'gene_id': int}).add_suffix('_A'),
            how='left', on=['gene_id_A']
        )
        .merge(
            df[['gene_id', 'cluster']].astype({'gene_id': int}).add_suffix('_B'),
            how='left', on=['gene_id_B']
        )
    )

    print('Calculating gene pair precision/recall metrics')

    # Calculate pairs with same annotation and same cluster
    df_gene_pair_same_annotation_same_cluster = (
        df_cluster_pairs
        .query(f'{annotation_name}_A=={annotation_name}_B')
        .drop_duplicates(['gene_id_A', 'gene_id_B'])
        .assign(same_annotation=True)
        [['gene_id_A', 'gene_id_B', 'same_annotation']]
    )

    df_cluster_pairs = (
        df_cluster_pairs
        .merge(df_gene_pair_same_annotation_same_cluster, how='left', on=['gene_id_A', 'gene_id_B'], validate='many_to_one')
    )

    results['gene_pair_same_annotation_same_cluster'] = df_gene_pair_same_annotation_same_cluster.pipe(len)

    results['gene_pair_different_annotation_same_cluster'] = (
        df_cluster_pairs
        .dropna(subset=[f'{annotation_name}_A', f'{annotation_name}_B'])
        .query('same_annotation != True')
        .drop_duplicates(['gene_id_A', 'gene_id_B'])
        .pipe(len)
    )
    
    # Calculate pairs with same annotation but different cluster
    df_annotation_pairs_cluster_['cluster_match'] = (
        df_annotation_pairs_cluster_['cluster_A'] == df_annotation_pairs_cluster_['cluster_B']
    )

    results['gene_pair_same_annotation_different_cluster'] = (
        (~df_annotation_pairs_cluster_.drop_duplicates(['gene_id_A', 'gene_id_B'])['cluster_match']).sum()
    )
    
    # Calculate precision and recall metrics
    try:
        results['gene_pair_precision'] = (
            results['gene_pair_same_annotation_same_cluster'] /
            (results['gene_pair_same_annotation_same_cluster'] + results['gene_pair_different_annotation_same_cluster'])
        )
    except:
        results['gene_pair_precision'] = np.nan

    results['gene_pair_recall'] = (
        results['gene_pair_same_annotation_same_cluster'] /
        (results['gene_pair_same_annotation_same_cluster'] + results['gene_pair_same_annotation_different_cluster'])
    )
    
    annotation_recall = (
        df_annotation_pairs_cluster_
        .groupby([annotation_name])
        .apply(lambda x: annotation_recall_pairs(x))
    )

    results['gene_pair_recall_annotation_mean'] = annotation_recall.mean()

    results['gene_pair_recall_annotation_weighted_mean'] = (
        (annotation_recall * df_annotation_pairs_cluster_[annotation_name].value_counts(normalize=True))
        .fillna(0)
        .sum()
    )

    annotation_precision = annotation_precision_pairs(
        df_cluster_pairs.dropna(subset=[f'{annotation_name}_A', f'{annotation_name}_B']),
        annotations=list(annotation_singles_match[annotation_name].unique())
    )
    results['gene_pair_precision_annotation_mean'] = annotation_precision.mean()

    results['gene_pair_precision_annotation_weighted_mean'] = (
        (annotation_precision * df_annotation_pairs_cluster_[annotation_name].value_counts(normalize=True))
        .fillna(0)
        .sum()
    )

    return results

def expand_ampersand_pairs(df, columns=['gene_symbol', 'gene_id']):
    """
    Expand rows where gene symbols contain '&' into multiple rows.
    
    Parameters:
        df (pd.DataFrame): DataFrame with gene pairs.
        columns (list): Columns to expand.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    keep = [col for col in df.columns if not any([b in col for b in columns])]
    df = df.reset_index(drop=True)  # Requires unique index
    df_ampersand = df[(df['gene_symbol_A'].str.contains('&') | df['gene_symbol_B'].str.contains('&'))]
    df_ = df.loc[~(df.index.isin(df_ampersand.index))]
    df_ampersand_A = split_concat_ampersand(df_ampersand, [col + '_A' for col in columns])
    df_ampersand_B = split_concat_ampersand(df_ampersand, [col + '_B' for col in columns])
    df_ampersand = df_ampersand_A.join(df_ampersand_B, how='outer').join(df_ampersand[keep], how='outer')
    df_ = pd.concat([df_, df_ampersand])
    return df_

def expand_ampersand_singles(df, columns=['gene_symbol', 'gene_id']):
    """
    Expand rows where gene symbols contain '&' into multiple rows for single gene DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with single gene annotations.
        columns (list): Columns to expand.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    keep = [col for col in df.columns if col not in columns]
    df = df.reset_index(drop=True)  # Requires unique index
    df_ampersand = df[(df['gene_symbol'].str.contains('&'))]
    df_ = df.loc[~(df.index.isin(df_ampersand.index))]
    df_ampersand_ = split_concat_ampersand(df_ampersand, columns)
    df_ampersand = df_ampersand_.join(df_ampersand[keep], how='outer')
    df_ = pd.concat([df_, df_ampersand])
    return df_

def split_concat_ampersand(df, columns):
    """
    Split gene symbols containing '&' and concatenate into separate rows.
    
    Parameters:
        df (pd.DataFrame): DataFrame with gene symbols.
        columns (list): Columns to split.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    return (
        pd.concat(
            [df[col].str.split('&', expand=True).stack().rename(col).droplevel(1) for col in columns],
            axis=1
        )
    )

def get_distance_pairs(distance_matrix, index):
    """
    Create a DataFrame of gene pairs with their distances from a distance matrix.
    
    Parameters:
        distance_matrix (np.ndarray): Matrix of distances.
        index (pd.Index): Index for rows and columns.

    Returns:
        pd.DataFrame: DataFrame with gene pairs and their distances.
    """
    count = min(distance_matrix.shape)
    df_pairs = pd.DataFrame(distance_matrix[np.triu_indices(count, k=1)],
                            index=pd.MultiIndex.from_tuples([
                                (*A, *B) for A, B in itertools.combinations(index, 2)],
                                names=[n + '_A' for n in index.names] + [n + '_B' for n in index.names]),
                            columns=['distance'])
    return df_pairs

def annotation_recall_pairs(df_annotation_pairs):
    """
    Calculate recall for annotation pairs.
    
    Parameters:
        df_annotation_pairs (pd.DataFrame): DataFrame with annotation pairs.

    Returns:
        float: Recall value.
    """
    if len(df_annotation_pairs) == 0:
        return 0.
    try:
        return df_annotation_pairs['cluster_match'].value_counts(normalize=True)[True]
    except:
        return 0.

def annotation_precision_pairs(df_cluster_pairs, annotations, annotation_name='complex'):
    """
    Calculate precision for annotation pairs.
    
    Parameters:
        df_cluster_pairs (pd.DataFrame): DataFrame with cluster pairs.
        annotations (list): List of unique annotations.
        annotation_name (str): Column name for the annotation.

    Returns:
        pd.Series: Precision values for each annotation.
    """
    df_cluster_pairs['annotation_match'] = (
        df_cluster_pairs[f'{annotation_name}_A'] == df_cluster_pairs[f'{annotation_name}_B']
    )
    d = dict()
    for annotation in annotations:
        df_a = df_cluster_pairs.query(f'({annotation_name}_A == @annotation) | ({annotation_name}_B == @annotation)')
        df_a = df_a.sort_values('annotation_match').drop_duplicates(subset=['gene_id_A', 'gene_id_B'], keep='last')
        try:
            precision = df_a['annotation_match'].value_counts(normalize=True)[True]
        except:
            precision = 0.
        d.update({annotation: precision})
    return pd.Series(d)
