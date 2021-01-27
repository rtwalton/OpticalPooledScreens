import numpy as np
import pandas as pd
from random import choice,choices

from ops.constants import *
from ops.utils import groupby_histogram, groupby_reduce_concat
from scipy.stats import wasserstein_distance, ks_2samp, ttest_ind, kstest

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdm_auto
from joblib import Parallel,delayed

def distribution_difference(df,col='dapi_gfp_corr',control_query='gene_symbol == "non-targeting"', groups='gene_symbol'):
    y_neg = (df
      .query(control_query)
      [col]
    )
    return df.groupby(groups).apply(lambda x:
      wasserstein_distance(x[col], y_neg))

def process_rep(df, value='dapi_gfp_corr_nuclear', 
               sgRNA_index=('sgRNA_name', 'gene_symbol')):
    """Calculate statistics for one replicate.
    
    Example:

    sample_index = ['replicate', 'stimulant', 'well']
    genes = ['MYD88', 'TRADD', 'nontargeting']
    stats = (df_cells
     .groupby(sample_index)
     .apply(process_rep).reset_index()
    )
    """
    sgRNA_index = list(sgRNA_index)
    nt = df.query('gene_symbol == "nontargeting"')[value]
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
    )

def get_simple_stats(df_stats):
    return (df_stats
     .groupby(['gene_symbol', 'stimulant'])
     .apply(lambda x: x.eval('mean * count').sum() / x['count'].sum())
     .rename('mean')
     .reset_index()
     .pivot_table(index='gene_symbol', columns='stimulant', values='mean')
     .assign(IL1b_rank=lambda x: x['IL1b'].rank().astype(int))
     .assign(TNFa_rank=lambda x: x['TNFa'].rank().astype(int))
    )

## BOOTSTRAPPING

def bootstrap_cells(s, n_cells=100, n_reps=10000, statistic=np.mean, n_jobs=-1, tqdm=False):
    vals = s.values
    def bootstrap(vals, n_cells,statistic):
        return statistic(choices(vals,k=n_cells,cum_weights=None))

    if tqdm:
        reps = tqdm_auto(range(n_reps))
    else:
        reps = range(n_reps)
    
    bootstrapped = Parallel(n_jobs=n_jobs)(delayed(bootstrap)(vals, n_cells, statistic) 
                                           for _ in reps)

    return np.array(bootstrapped)

def bootstrap_within_guides(s, n_cells=100, n_reps=10000, statistic=np.mean, n_jobs=-1, tqdm=False):
    guide_values = {k:g.values for k,g in s.groupby('sgRNA')}
    guides = list(guide_values)

    if tqdm:
        reps = tqdm_auto(range(n_reps))
    else:
        reps = range(n_reps)

    def bootstrap(guide_values,guides,n_cells,statistic):
        rep_guide = choice(guides)
        return statistic(choices(guide_values[rep_guide], k=n_cells, cum_weights=None))
    
    bootstrapped = Parallel(n_jobs=n_jobs)(delayed(bootstrap)(guide_values,guides,n_cells,statistic) 
                                           for _ in reps)

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
        return (bootstrapped_nt>measured).mean(), (bootstrapped_nt<measured).mean()
    else:
        raise ValueError(f'tails=={tails} not implemented')

def bootstrap_gene_pval(s_targeting_guide_scores,guide_null_distributions,gene_statistic=np.median,n_reps=10000):
    """`guide_null_distributions` is of shape (n_guides,n_reps_guide_bootstrapping), e.g., a different null
    distribution for each guide based on its sample size"""
    measured = gene_statistic(s_targeting_guide_scores)
    guides = len(s_targeting_guide_scores)
    
    samples = np.random.randint(guide_null_distributions.shape[1],size=(guides,n_reps))
    gene_null = gene_statistic(guide_null_distributions[np.array([n for n in range(guides)]).reshape(-1,1),samples],axis=0)
    
    return max(min((gene_null>measured).mean(),(gene_null<measured).mean()),1/n_reps)*2

## PLOTTING

def plot_distributions(df_cells, gene, col='dapi_gfp_corr_nuclear',
    control_query='gene_symbol=="nt"', replicate_col='replicate', conditions_col='stimulant',
    conditions = ['TNFa', 'IL1b'], range=(-1,1), n_bins=100
    ):
    
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


# OLD (pre-binned)

def cells_to_distributions(df_cells, bins, column='dapi_gfp_corr_nuclear'):
    """
    
    Make sure to .dropna() first.
    """
    index = [GENE_SYMBOL, SGRNA_NAME, REPLICATE, STIMULANT]
    return (df_cells
     .pipe(groupby_histogram, index, column, bins)
     )


def plot_distributions_old(df_dist):
    """Old plotting function. 
    Plots from data that is already binned. Pre-filter for gene symbol of
    interest and LG non-targeting guides (shown individually).
    """

    # sgRNA names
    hue_order = (df_dist.reset_index()['sgRNA_name'].value_counts()
        .pipe(lambda x: natsorted(set(x.index))))
    colors = iter(sns.color_palette(n_colors=10))
    palette, legend_data = [], {}
    for name in hue_order:
        palette += ['black' if name.startswith('LG') else colors.next()]
        legend_data[name] = patches.Patch(color=palette[-1], label=name)


    
    def plot_lines(**kwargs):
        df = kwargs.pop('data')
        color = kwargs.pop('color')
        ax = plt.gca()
        (df
         .filter(regex='\d')
         .T.plot(ax=ax, color=color)
        )

    fg = (df_dist
     .pipe(normalized_cdf)
     .reset_index()
     .pipe(sns.FacetGrid, row='stimulant', hue='sgRNA_name', col='replicate', 
           palette=palette, hue_order=hue_order)
     .map_dataframe(plot_lines)
     .set_titles("{row_name} rep. {col_name}")
     .add_legend(legend_data=legend_data)
    )
    return fg

def generalized_log(y,offset=0):
    return np.log((y + np.sqrt(y**2 + offset))/2)

def feature_normality_test(df,columns='all'):
    """tests for normality of feature distributions using the KS-test
    """
    if columns == 'all':
        columns = df.columns
        
    results = []
    
    for col in columns:
        values=df[col].values
        standardized = (values-values.mean())/values.std()
        ks_result = kstest(standardized,'norm')
        results.append({'feature':col,'ks_statistic':ks_result[0],'p_value':ks_result[1]})
        
    return pd.DataFrame(results)
