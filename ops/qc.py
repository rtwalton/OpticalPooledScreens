import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def threshold_vs_mapping(df_mapped_reads,threshold_var='peak',ax=None):
    mapping_rate =[]
    spots_mapped = []
    if df_mapped_reads[threshold_var].max()<100:
        thresholds = np.array(range(0,int(np.quantile(df_mapped_reads[threshold_var],q=0.99)*1000)))/1000
    else:
        thresholds = list(range(0,int(np.quantile(df_mapped_reads[threshold_var],q=0.99)),10))
    df_passed = df_mapped_reads.query('cell>0')
    for threshold in thresholds:
        df_passed = df_passed.query('{} > @threshold'.format(threshold_var))
        spots_mapped.append(df_passed.dropna(subset=['gene_symbol']).pipe(len))
        mapping_rate.append(df_passed.dropna(subset=['gene_symbol']).pipe(len)/df_passed.pipe(len))
    
    if not ax:
        ax = sns.lineplot(x = thresholds, y = mapping_rate);
    else:
        sns.lineplot(x = thresholds, y = mapping_rate, ax=ax);
    plt.ylabel('mapping rate');
    plt.xlabel('{} threshold'.format(threshold_var));
    ax_right = ax.twinx()
    sns.lineplot(x = thresholds, y = spots_mapped, ax = ax_right, color = 'coral')
    plt.ylabel('mapped spots');
    plt.legend(ax.get_lines()+ax_right.get_lines(),['mapping rate','# mapped spots'],loc=7);
    return thresholds, mapping_rate, spots_mapped