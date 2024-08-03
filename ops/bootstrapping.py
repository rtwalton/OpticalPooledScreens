import numpy as np
import pandas as pd
from random import choices
from sklearn.metrics import roc_auc_score

def feature_transform(df, transformation_dict, channels):
    def apply_transformation(feature, transformation):
        if transformation == 'log(feature)':
            return np.log(feature)
        elif transformation == 'log(feature-1)':
            return np.log(feature - 1)
        elif transformation == 'log(1-feature)':
            return np.log(1 - feature)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    # Apply each transformation to the corresponding feature in the dataframe
    for _, row in transformation_dict.iterrows():
        feature_template = row['feature']
        transformation = row['transformation']
        
        # Handle single channel features
        if '{channel}' in feature_template:
            for channel in channels:
                feature = feature_template.replace("{channel}", channel)
                if feature in df.columns:
                    # Apply the transformation
                    df[feature] = apply_transformation(df[feature], transformation)
        
        # Handle double channel features (overlap)
        elif '{channel1}' in feature_template and '{channel2}' in feature_template:
            for channel1 in channels:
                for channel2 in channels:
                    if channel1 != channel2:
                        feature = feature_template.replace("{channel1}", channel1).replace("{channel2}", channel2)
                        if feature in df.columns:
                            # Apply the transformation
                            df[feature] = apply_transformation(df[feature], transformation)
    
    return df

# Function to 
def grouped_standardization(df, population_feature='gene_symbol', control_prefix='nontargeting', group_columns=['plate', 'well'], index_columns=['tile', 'cell'], cat_columns=['gene_symbol', 'sgRNA'], target_features=None, drop_features=False):
    '''Standardizes the numerical columns of df by evaluating the robust z-score. The null model for each
    measurement is estimated as its empirical distribution for the control_prefix. If group_column is specified, the 
    null model is evaluated separately for each category in group_column. (E.g., standardizing by well.)'''

    # Warning, this will fail if dataframe contains repeated values for cells
    df_out = df.copy().drop_duplicates(subset=group_columns + index_columns)

    if target_features is None:
        target_features = [col for col in df.columns if col not in group_columns + index_columns + cat_columns]
    
    if drop_features:
        df = df[group_columns + index_columns + cat_columns + target_features]

    unstandardized_features = [col for col in df.columns if col not in target_features]
    
    # Filter the control group based on control_prefix
    control_group = df[df[population_feature].str.startswith(control_prefix)]
    
    # Define a custom function to calculate MAD
    def median_absolute_deviation(arr):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return mad
    
    # Calculate group medians and MADs for the control group
    group_medians = control_group.groupby(group_columns)[target_features].median()
    group_mads = control_group.groupby(group_columns)[target_features].apply(lambda x: x.apply(median_absolute_deviation))

    df_out = pd.concat([df_out[unstandardized_features].set_index(group_columns+index_columns),
                        df_out.set_index(group_columns+index_columns)[target_features].subtract(group_medians).divide(group_mads).multiply(0.6745)],
                       axis=1)

    return df_out.reset_index()

# Classify into mitotic and interphase cells
def split_mitotic_simple(df, conditions):
    """
    Split the DataFrame into mitotic and interphase based on given conditions.

    Parameters:
    - df: pd.DataFrame, the DataFrame to be split.
    - conditions: dict, a dictionary where keys are feature names and values are tuples
                  in the form (cutoff, direction) where direction is 'greater' or 'less'.
    
    Returns:
    - mitotic_df: pd.DataFrame, DataFrame containing rows that meet all conditions.
    - interphase_df: pd.DataFrame, DataFrame containing rows that do not meet all conditions.
    """
    # Initialize the mitotic DataFrame with all rows from the original DataFrame
    mitotic_df = df.copy()
    
    # Iterate over conditions and filter the mitotic DataFrame
    for feature, (cutoff, direction) in conditions.items():
        if direction == 'greater':
            mitotic_df = mitotic_df[mitotic_df[feature] > cutoff]
        elif direction == 'less':
            mitotic_df = mitotic_df[mitotic_df[feature] < cutoff]
        else:
            raise ValueError("Direction must be 'greater' or 'less'")
    
    # The interphase DataFrame contains rows not in the mitotic DataFrame
    interphase_df = df.drop(mitotic_df.index)
    
    return mitotic_df, interphase_df

def collapse_to_sgrna(df, method='median', target_features=None, index_features=['gene_symbol', 'sgRNA'], control_prefix='nontargeting', min_count=None):
    """
    Collapse cells to the sgRNA level.

    Parameters:
    df (pd.DataFrame): Input dataframe containing cell-level data.
    method (str): Method to use for collapsing ('median' or 'auc').
    target_features (list): List of feature columns to compute the score for.
    control_prefix (str): Prefix in gene_col that represents the non-targeting control.
    index_features (list): List of columns to group by for sgRNA and control population.
    min_count (int): Minimum number of cells per sgRNA to include in the results.

    Returns:
    pd.DataFrame: Dataframe with sgRNA scores for all features.
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    if method == 'median':
        df_out = df.groupby(index_features)[target_features].median().reset_index()
        df_out['sgrna_count'] = df.groupby(index_features).size().reset_index(name='sgrna_count')['sgrna_count']
        if min_count is not None:
            df_out = df_out.query('sgrna_count >= @min_count')
        return df_out

    elif method == 'auc': # NEED TO FIX
        raise NotImplementedError('Have not implemented deltaAUC as a variate yet.')

#         # Get the non-targeting control population
#         nt_df = df[df['sgRNA'].str.startswith(control_prefix)]

#         sgrna_scores_list = []
#         for feature in target_features:
#             # Compute the AUC for each sgRNA
#             def compute_auc(sub_df):
#                 labels = [1] * len(sub_df) + [0] * len(nt_df)
#                 scores = list(sub_df[feature]) + list(nt_df[feature])
#                 return roc_auc_score(labels, scores)

#             sgrna_scores = df.groupby(index_features[1]).apply(compute_auc).reset_index()
#             sgrna_scores.columns = [index_features[1], 'score']

#             # Compute the AUC for the non-targeting control population
#             nt_labels = [1] * len(nt_df) + [0] * len(nt_df)
#             nt_scores = list(nt_df[feature]) + list(nt_df[feature])
#             nt_auc = roc_auc_score(nt_labels, nt_scores)

#             # Subtract the nt AUC from each sgRNA AUC
#             sgrna_scores['score'] = sgrna_scores['score'] - nt_auc
#             sgrna_scores['feature'] = feature
#             sgrna_scores_list.append(sgrna_scores)

#         # Combine all features into one DataFrame
#         result = pd.concat(sgrna_scores_list, axis=0)

#         # Add population count
#         result['population_count'] = df.groupby(index_features)[target_features].count().reset_index(drop=True)

#         if min_count is not None:
#             result = result.query('population_count >= @min_count')

#         return result

    else:
        raise ValueError("Method must be either 'median' or 'auc'")
        
def collapse_to_gene(df, target_features=None, index_features=['gene_symbol'], min_count=None):
    """
    Collapse cells to the sgRNA level.

    Parameters:
    df (pd.DataFrame): Input dataframe containing cell-level data.
    method (str): Method to use for collapsing ('median' or 'auc').
    target_features (list): List of feature columns to compute the score for.
    control_prefix (str): Prefix in gene_col that represents the non-targeting control.
    index_features (list): List of columns to group by for sgRNA and control population.
    min_count (int): Minimum number of cells per sgRNA to include in the results.

    Returns:
    pd.DataFrame: Dataframe with sgRNA scores for all features.
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    df_out = df.groupby(index_features)[target_features].median().reset_index()

    if 'sgrna_count' in df.columns:
        df_out['gene_count'] = df.groupby(index_features)['sgrna_count'].sum().reset_index(drop=True)

    if min_count is not None:
        df_out = df_out.query('gene_count >= @min_count')
        
    return df_out

def split_null(df, control_feature='gene_symbol', control_prefix='nontargeting'):
    """
    Split the DataFrame into a control group based on specified feature and prefix.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    control_feature (str): Column name to check for control prefixes.
    control_prefix (str): The prefix in the control_feature to filter for.

    Returns:
    pd.DataFrame: DataFrame containing only the control group.
    """
    control_group = df[df[control_feature].str.startswith(control_prefix)]
    return control_group

def split_by_population(df,population_values,population_feature='cell_barcode_0'):
    return [df.query(population_feature+'==@population') for population in population_values]

def compute_variate(df, variate='median', target_features=None,
                    index_features=['gene_symbol', 'cell_barcode_0'],
                    n_samples=10000, sample_depth=None,
                    min_count=None):
    """
    Compute variates with bootstrapping.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    variate (str): Variate to compute ('median' or 'auc').
    target_features (list): List of feature columns to compute the score for.
    index_features (list): List of columns to group by.
    n_samples (int): Number of times to sample variate per index_feature group.
    sample_depth (int): Number of cells to sample for variate computation.
    min_count (int): Minimum number of cells per index group to include in the results.

    Returns:
    pd.DataFrame: Dataframe with computed variates.
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    if variate == 'median':
        if sample_depth is not None: 
            df_out = pd.concat([df.groupby(index_features).sample(sample_depth, replace=True)
                                .groupby(index_features)[target_features].median()
                                .reset_index() for _ in range(n_samples)
                                ], axis=0).assign(population_count=sample_depth)
            return df_out
        else:
            raise NotImplementedError('Bootstrapping without subsampling is not yet implemented')

    if variate == 'auc':
        raise NotImplementedError('Have not implemented deltaAUC as a variate yet.')
        
def evaluate_p_value(df_exp, df_boot, target_features, population_columns, two_sided=True):
    df_exp_population = df_exp[population_columns]
    df_exp = df_exp[target_features]
    df_boot = df_boot[target_features]

    def _get_p(series):
        if two_sided:
            return 2 * pd.concat([(df_boot > series).sum(), (df_boot < series).sum()], axis=1).transpose().min() / len(df_boot)
        else:
            return (df_boot > series).sum().transpose() / len(df_boot)
        
    p_values = pd.concat([_get_p(df) for _, df in df_exp.iterrows()], axis=1).transpose().rename(columns=lambda x: 'p_' + x)
    
    # Add the population columns and reorder
    for col in reversed(population_columns):
        p_values.insert(0, col, df_exp_population[col].values)
    
    return p_values

        
        
        
        
        
        
        
        
        
        
        
        
    
def add_pseudogenes(df,gene_feature='gene_symbol',guide_feature='cell_barcode_0',pseudogene_population='nontargeting',sample_depth=None,guides_per_pseudogene=4):
    ''' Duplicates the non-targeting cells and groups them into "pseudo-genes" to serve as a non-targeting control for gene level volcano plots.
    '''
    guide_list = df.query(gene_feature+'==@pseudogene_population')[guide_feature].unique()
    num_guides = len(guide_list)
    gene_symbol = []
    for count in range(int(num_guides/guides_per_pseudogene+1)):
        gene_symbol.append(guides_per_pseudogene*[f'pseudogene{count}'])
    gene_symbol = np.concatenate(gene_symbol)
    
    df_pseudo = df.query(gene_feature+'==@pseudogene_population')
    conversion_dict = {guide:gene for guide,gene in zip(guide_list,gene_symbol[:num_guides])}
    df_pseudo[gene_feature] = df_pseudo[guide_feature].apply(lambda x: conversion_dict[x])
    return pd.concat([df,df_pseudo],axis=0)

def merge_composite_variate_bootstrap(df_list,how='median',wildcards=None,drop_columns=[]):
    from random import sample
    population = None
    if wildcards is not None:
        if 'population' in wildcards.keys():
            population = wildcards.population

    column_names = df_list[0].drop(columns=drop_columns).columns

    return pd.DataFrame(np.median([sample(df.drop(columns=drop_columns).values.tolist(),k=len(df)) for df in df_list],axis=0),columns=column_names).assign(population=population).assign(population_count=wildcards.depth_string)
    #return pd.DataFrame(np.median([df.drop(columns=drop_columns).values for df in df_list],axis=0),columns=column_names).assign(population=population).assign(population_count=wildcards.depth_string)

def collect_volcano(df_list_var,df_list_pval,left_population_features=['population'],right_population_features=['population']):
    '''Used for collecting multiple dataframes, and merging them into a single output. 
    df_list_var: a list of DataFrames with computed variates
    df_list_pval: a list of DataFames with computed p-vals 
    population_features: list of features in the two sets of DataFrames that will be used to align the data.
    '''
    print(len(df_list_var))
    print(len(df_list_pval))

    return pd.concat(df_list_var,axis=0).merge(pd.concat(df_list_pval,axis=0),left_on=left_population_features,right_on=right_population_features)




## BOOTSTRAPPING

def bootstrap_cells(s, n_cells=100, n_reps=10000, statistic=np.mean, n_jobs=1, tqdm=False):
    rng = np.random.default_rng()
    vals = s.values
    def bootstrap(vals, n_cells,statistic):
        return statistic(vals[rng.integers(len(vals),size=n_cells)])

    if tqdm:
        reps = tqdm_auto(range(n_reps))
    else:
        reps = range(n_reps)
    
    if n_jobs!=1:
        bootstrapped = Parallel(n_jobs=n_jobs)(delayed(bootstrap)(vals, n_cells, statistic) 
                                               for _ in reps)
    else:
        bootstrapped = [bootstrap(vals, n_cells, statistic) for _ in reps]

    return np.array(bootstrapped)

def bootstrap_within_guides(s, n_cells=100, n_reps=10000, statistic=np.mean, n_jobs=1, tqdm=False):
    rng = np.random.default_rng()
    guide_values = {k:g.values for k,g in s.groupby('sgRNA')}
    guides = list(guide_values)

    if tqdm:
        reps = tqdm_auto(range(n_reps))
    else:
        reps = range(n_reps)

    def bootstrap(guide_values,guides,n_cells,statistic):
        rep_guide = rng.choice(guides)
        vals = guide_values[rep_guide]
        return statistic(vals[rng.integers(len(vals), size=n_cells)])
    
    if n_jobs!=1:
        bootstrapped = Parallel(n_jobs=n_jobs)(delayed(bootstrap)(guide_values,guides,n_cells,statistic) 
                                               for _ in reps)
    else:
        bootstrapped = [bootstrap(guide_values,guides,n_cells,statistic) for _ in reps]

    return np.array(bootstrapped)

def bootstrap_guide_pval(s_nt, s_targeting, n_reps=10000, statistic=np.mean, bootstrap_nt_within_guides=True, 
    tails='two', n_jobs=-1, tqdm=False):
    n_cells = s_targeting.pipe(len)
    measured = statistic(s_targeting)

    if bootstrap_nt_within_guides:
        bootstrap = bootstrap_within_guides
    else:
        bootstrap = bootstrap_cells

    bootstrapped_nt = bootstrap(s_nt, n_cells, n_reps=n_reps, statistic=statistic, n_jobs=n_jobs, tqdm=tqdm)
    
    if tails=='two':
        return max(min((bootstrapped_nt>measured).mean(),(bootstrapped_nt<measured).mean()),1/n_reps)*2
    elif tails=='one':
        return min((bootstrapped_nt>measured).mean(), (bootstrapped_nt<measured).mean())
    else:
        raise ValueError(f'tails=={tails} not implemented')

def bootstrap_gene_pval(s_targeting_guide_scores, guide_null_distributions, gene_statistic=np.median,
    n_reps=10000, tails='two'):
    """`guide_null_distributions` is of shape (n_guides,n_reps_guide_bootstrapping), e.g., a different null
    distribution for each guide based on its sample size"""
    rng = np.random.default_rng()
    measured = gene_statistic(s_targeting_guide_scores)
    
    gene_null = gene_statistic(rng.choice(guide_null_distributions,size=n_reps,replace=True,axis=1),axis=0)
    
    if tails=='two':
        return max(min((gene_null>measured).mean(),(gene_null<measured).mean()),1/n_reps)*2
    elif tails=='one':
        return min((gene_null>measured).mean(),(gene_null<measured).mean())
    else:
        raise ValueError(f'tails=={tails} not implemented')
