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

def distribution_difference(df,col='dapi_gfp_corr',control_query='gene_symbol == "non-targeting"', groups='gene_symbol'):
    y_neg = (df
      .query(control_query)
      [col]
    )
    return df.groupby(groups).apply(lambda x:
      wasserstein_distance(x[col], y_neg))

def process_rep(df, value='dapi_gfp_corr_nuclear', 
               sgRNA_index=('sgRNA_name', 'gene_symbol'),
               control_query='gene_symbol=="nontargeting"'
               ):
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
    return (df_stats
     .groupby(['gene_symbol', 'stimulant'])
     .apply(lambda x: x.eval('mean * count').sum() / x['count'].sum())
     .rename('mean')
     .reset_index()
     .pivot_table(index='gene_symbol', columns='stimulant', values='mean')
     .assign(IL1b_rank=lambda x: x['IL1b'].rank().astype(int))
     .assign(TNFa_rank=lambda x: x['TNFa'].rank().astype(int))
    )

def enrichment_score(sorted_metric,ranks,p=1,return_running=False):
    """`sorted_metric` should be pre-sorted, and `ranks` should be computed with `method`='first'
    """
    P = np.ones(len(sorted_metric))*(-1/(len(sorted_metric)-len(ranks)))
    
    N_R = (abs(sorted_metric[ranks])**p).sum()
    P[ranks] = abs(sorted_metric[ranks])**p/N_R
    
    P_running = P.cumsum()
    
    leading_edge = np.argmax(abs(P_running))
    
    if return_running:
        return P_running,leading_edge
    return P_running[leading_edge]

def enrichment_score_minimal(selected_metric,selected_ranks,N,p=1):
    """Calculate enrichment score only using the minimal information necessary (ranks and values for the 
    selected set, total number of genes)
    """
    P = np.ones(N)*(-1/(N-len(selected_ranks)))
    
    N_R = (abs(selected_metric)**p).sum()
    P[sorted(selected_ranks)] = (np.abs(sorted(selected_metric,reverse=True))**p)/N_R
    
    P_running = P.cumsum()
    
    leading_edge = np.argmax(abs(P_running))
    
    return P_running[leading_edge]

def aggregate_enrichment_score(df,grouping,cols=None,p=1):
    # this is slow but it works. could parallelize per-column or per-group scoring.
    if cols is None:
        cols = df.columns
    df = pd.concat([df,df[cols].rank(ascending=False,method='first').astype(int).add_suffix('_rank')-1],axis=1)
    
#     return df
    N = df.pipe(len)
    
    arr = []
    for col in tqdm(cols):
        arr.append(df
                   .groupby(grouping)
                   [[col,f'{col}_rank']]
                   .apply(lambda x: enrichment_score_minimal(x.iloc[:,0].values,x.iloc[:,1].values,N,p=p))
                   .rename(col)
                  )
    return pd.concat(arr,axis=1)

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
    
    if n_job!=1:
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

## FEATURE PROCESSING

def generalized_log(y,offset=0):
    return np.log((y + np.sqrt(y**2 + offset**2))/2)

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

def get_feature_pair_correlations(df_corr):
    return (pd.DataFrame(
        df_corr.values[np.triu(np.ones(df_corr.shape,dtype=bool),k=1)],columns=['correlation'])
        .assign(**{f'feature_{n}':features 
            for n,features in enumerate(zip(*tuple(combinations(df_corr.columns,2))))})
        )

def get_feature_correlation_connected_components(df_corr,threshold=0.9):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    graph = csr_matrix(((df_corr.abs()>threshold).values-np.eye(df_corr.pipe(len))))
    n_components,labels = connected_components(csgraph=graph,directed=False,return_labels=True)

    return pd.DataFrame({'feature':df_corr.columns,'component':labels})

def get_feature_pair_info(df,threshold=0.9):
    df_corr = df.corr()

    df_pairs = get_feature_pair_correlations(df_corr)
    df_components = get_feature_correlation_connected_components(df_corr,threshold=threshold)

    return (df_pairs
        .merge(df_components.add_suffix('_0'),how='left',on='feature_0')
        .merge(df_components.add_suffix('_1'),how='left',on='feature_1')
        )

def visualize_connected_components(df,threshold=0.9,scale=0.5):
    import networkx as nx 
    df_corr = df.corr().abs()
    df_corr[df_corr<threshold] = 0
    G = nx.from_numpy_matrix((df_corr.values-np.eye(len(df.columns))))

    positions = nx.spring_layout(G,scale=scale)
    position_values = np.array(list(positions.values()))
    x_max,y_max = position_values.max(axis=0)
    x_min,y_min = position_values.min(axis=0)
    x_margin = (x_max-x_min)*0.25
    y_margin = (y_max-y_min)*0.25

    labels = '\n'.join([f'{node}: {label}' for label,node in zip(df.columns,positions.keys())])

    label_numbers  = {node:node for node in positions.keys()}

    fig,ax = plt.subplots(1,2,figsize=(10,5))

    nx.draw_networkx(G,pos=positions,labels=label_numbers,node_size=50,ax=ax[0])

    ax[0].set_xlim(x_min-x_margin,x_max+x_margin)
    ax[0].set_ylim(y_min-y_margin,y_max+y_margin)
    ax[0].axis('on')
    ax[1].text(0, 0.5,labels,horizontalalignment='left',verticalalignment='center',transform = ax[1].transAxes)
    ax[1].axis('off')
    return G,positions,fig,ax