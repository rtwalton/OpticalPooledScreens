import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
import pandas as pd
from ops.io import GLASBEY
import random
from itertools import combinations
from functools import partial
from adjustText import adjust_text

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.size'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def volcano(
    df,
    feature=None,
    x="score",
    y="pval",
    alpha_level=0.05,
    change=0,
    annotate=None,
    prelog=True,
    xscale=None,
    control_query=None,
    high_color='green',
    low_color='magenta',
    default_color='gray',
    threshold_kwargs=dict(color="gray", linestyle="--"),
    control_kwargs=dict(color=sns.color_palette()[1],label="non-targeting guides"),
    ax=None,
    rasterized=True,
    **kwargs
):

    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # `feature` is not None, `x` and `y` are suffices for paired columns
    if feature is not None:
        x = f"{feature}_{x}".strip("_")
        y = f"{feature}_{y}".strip("_")

    # `alpha_level` is None, nothing is marked as significant
    if alpha_level is None:
        alpha_level = df_[y].min() - 0.1

    # `change` is a single value, make cutoffs symmetric around 0
    if not isinstance(change, list):
        try:
            change = [-change, change]

        # negative fails for None
        except:
            change = [change, change]

    # None in `change` -> nothing marked marked as significant in that direction
    # if both elements of `change` are None, everything marked as significant
    if change == [None, None]:
        change = [
            df_[x].min() - 1,
        ] * 2
    else:
        if change[0] is not None:
            ax.axvline(change[0], **threshold_kwargs)
        else:
            change[0] = df_[x].min() - 1

        if change[1] is not None:
            ax.axvline(change[1], **threshold_kwargs)
        else:
            change[1] = df_[x].max() + 1

    df_["significant"] = False
    df_.loc[(df_[x] < change[0]) & (df_[y] < alpha_level), "significant"] = "low"
    df_.loc[(df_[x] > change[1]) & (df_[y] < alpha_level), "significant"] = "high"

    if prelog:
        df_[y] = -np.log10(df_[y])
        alpha_level = -np.log10(alpha_level)

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    sns.scatterplot(
        data=df_,
        x=x,
        y=y,
        hue="significant",
        hue_order=["high", False, "low"],
        palette=[high_color, default_color, low_color],
        legend=None,
        ax=ax,
        rasterized=rasterized,
        **kwargs
    )

    if control_query is not None:
        kwargs.update(control_kwargs)
        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs,
            # **control_kwargs,
        )

    # sns.scatterplot(data=df_.query('gene_symbol==@annotate_labels'),x=x,
    #                 y=y,hue='significant',hue_order=['high','low'],palette=['green','magenta'],edgecolor='black',ax=ax)

    if not prelog:
        ax.set_yscale("log", basey=10)
        y_max = np.floor(np.log10(df_[y].min()))
        ax.set_ylim([1.15, 10 ** (y_max)])
        ax.set_yticks(np.logspace(y_max, 0, -(int(y_max) - 1)))
        ax.set_yticklabels(
            labels=[
                f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                for n in np.linspace(y_max, 0, -(int(y_max) - 1))
            ]
        )

    if xscale == "symlog":
        ax = symlog_axis(df_[x],ax,'x')

    if annotate is not None:
	    for gene,data in df_.query('significant=="low"').nlargest(annotate,y).iterrows():
	    	ax.annotate(gene,(data[x],data[y]),
	    		textcoords='offset points',
	    		ha='right',
	    		xytext=(-5,0)
	    		)
	    for gene,data in df_.query('significant=="high"').nlargest(annotate,y).iterrows():
	    	ax.annotate(gene,(data[x],data[y]),
	    		textcoords='offset points',
	    		ha='left',
	    		xytext=(5,0)
	    		)

    # for y_offset,alignment,label in annotate:
    #     s = df_.query('gene_symbol==@label').iloc[0]
    #     ax.text(s[x],s[y]+y_offset,
    #             label,horizontalalignment=alignment,fontsize=12)

    ax.axhline(alpha_level, **threshold_kwargs)

    ax.set_xlabel(" ".join(x.split("_")))
    if prelog:
        ax.set_ylabel("-log10(p-value)")
    else:
        ax.set_ylabel("p-value")

    return ax

def two_feature(
    df,
    x,
    y,
    annotate_query=None,
    annotate_labels=False,
    annotate_kwargs=dict(edgecolor='black'),
    xscale=None,
    yscale=None,
    control_query=None,
    control_kwargs=dict(color=sns.color_palette()[1],label="non-targeting guides"),
    ax=None,
    rasterized=True,
    **kwargs
):

    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if annotate_query is not None:
        df_annotate = df_.query(annotate_query)

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    sns.scatterplot(
        data=df_,
        x=x,
        y=y,
        legend=None,
        ax=ax,
        rasterized=rasterized,
        **kwargs
    )

    if control_query is not None:
        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)
        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )

    if annotate_query is not None:
        kwargs_ = kwargs.copy()
        kwargs_.update(annotate_kwargs)
        sns.scatterplot(
            data=df_annotate,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )
        if annotate_labels:
            labels = []
            for _,entry in df_annotate.iterrows():
                labels.append(ax.annotate(entry[annotate_labels],(entry[x],entry[y]),arrowprops=dict(arrowstyle='-',relpos=(0,0),shrinkA=0,shrinkB=0)))


    if xscale == "symlog":
        ax = symlog_axis(df_[x],ax,'x')

    if yscale == "symlog":
        ax = symlog_axis(df_[y],ax,'y')

    ax.set_xlabel(" ".join(x.split("_")))
    ax.set_ylabel(" ".join(y.split("_")))

    try:
        adjust_text(labels,df_[x].values,df_[y].values)
    except:
        pass

    return ax


def dimensionality_reduction(
    df,
    x="X",
    y="Y",
    default_kwargs={'color':'lightgray','alpha':0.5},
    control_query='gene_id=="-1"',
    control_color="black",
    control_legend=True,
    control_kwargs=dict(),
    label_query=None,
    label_hue="cluster",
    label_as_cmap=False,
    label_palette="glasbey",
    label_kwargs=dict(),
    randomize_palette=False,
    label_legend=False,
    legend_kwargs=dict(),
    hide_axes=False,
    ax=None,
    rasterized=True,
    **kwargs,
):

    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    if label_query is not None:
        df_label = df_.query(label_query)
        df_ = df_[~(df_.index.isin(df_label.index))]

    sns.scatterplot(data=df_, x=x, y=y, **default_kwargs, **kwargs, ax=ax, rasterized=rasterized)

    if control_query is not None:
        # control_kwargs = dict()
        if "legend" not in control_kwargs:
	        if isinstance(control_legend, str):
	            control_kwargs["label"] = control_legend
	        elif control_legend:
	            control_kwargs["label"] = "control"
	        else:
	            control_kwargs["legend"] = False
        
        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)

        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            color=control_color,
            alpha=0.75,
            **kwargs_,
            ax=ax,
            rasterized=rasterized
        )

    if label_query is not None:
        if label_hue is not None:
            n_colors = df_label[label_hue].nunique()

            if label_palette == "glasbey":
                palette = sns.color_palette(
                    ((GLASBEY.reshape(3, 256).T) / 256)[1:], n_colors=n_colors
                )
            else:
                palette = sns.color_palette(
                    label_palette, n_colors=n_colors, as_cmap=label_as_cmap
                )

            if randomize_palette:
                random.seed(int(randomize_palette))
                random.shuffle(palette)
        else:
        	palette = None

        kwargs_ = kwargs.copy()
        kwargs_.update(label_kwargs)

        sns.scatterplot(
            data=df_label,
            x=x,
            y=y,
            hue=label_hue,
            palette=palette,
            legend=label_legend,
            **kwargs_,
            ax=ax,
            rasterized=rasterized
        )

        if label_legend:
            if label_as_cmap:
                hue_norm = kwargs.get(
                    'hue_norm',
                    (df_label[label_hue].astype(float).min(),df_label[label_hue].astype(float).max())
                    )
                s = kwargs.get('s',10)
                hdl,_ = ax.get_legend_handles_labels()
                legend_colors = sns.color_palette(label_palette,as_cmap=True)(np.linspace(0,255,5,dtype=int))
                legend_color_vals = np.linspace(*hue_norm,5)
                legend_color_header=hdl[0]
                legend_elements = [
                    plt.scatter([],[],marker='o',s=s,color=c,
                        linewidth=0.5,edgecolor='k',label=str(cl)) 
                    for c,cl in zip(legend_colors,legend_color_vals)]
                ax.legend(handles=legend_elements, loc=(1.05,0.33), ncol=1, **legend_kwargs)
            else:
                n_cols=max(1, (n_colors // 20))
                ax.legend(loc=(1, 0.33), ncol=n_cols,**legend_kwargs)

    if hide_axes:
        ax.axis("off")

    return ax

def heatmap(
    df,
    figsize=None,
    row_colors=None,
    label_fontsize=5,
    rasterized=True,
    dendrogram_ratio=0,
    colors_ratio=0.1,
    spinewidth=0.25,
    alternate_ticks=(True,True),
    alternate_tick_length=(30,30),
    label_every=(1,1),
    xticklabel_kwargs=dict(),
    yticklabel_kwargs=dict(),
    **kwargs
    ):
    """Note: weird things happen if you make the heatmap aspect ratio too big/small 
    (e.g., figsize=(1.3,6) looks about the same as (2.7,6)).
    """

    if isinstance(row_colors,str):
        # assume is a column name
        row_color_map = {group:color 
            for group,color 
            in zip(df[row_colors].unique(),sns.color_palette('Set2',n_colors=df[row_colors].nunique()))}
        row_colors = df[row_colors].map(row_color_map)

    if row_colors is not None:
        df = df[[col for col in df.columns if col!=row_colors.name]]

    y_len,x_len = df.shape

    if figsize is None:
        figsize = np.array([x_len,y_len])*0.08
        print(figsize)
    print(figsize)
    print(x_len,y_len)
    
    vmin = kwargs.pop('vmin',-5)
    vmax = kwargs.pop('vmax',5)
    col_cluster = kwargs.pop('col_cluster',False)
    row_cluster = kwargs.pop('row_cluster',False)
    cmap = kwargs.pop('cmap','vlag')
    cbar_pos = kwargs.pop('cbar_pos',None)

    cg = sns.clustermap(
        df,
        figsize=figsize,
        row_colors=row_colors,
        vmin=vmin,
        vmax=vmax,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        cmap=cmap,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        rasterized=rasterized,
        cbar_pos=cbar_pos,
        **kwargs
    )

    cg.gs.update(hspace=0,wspace=0) # remove space between subplots
    # remove color axis ticks
    if row_colors is not None:
    # cg.ax_col_colors.set_yticks([])
        cg.ax_row_colors.set_xticks([])

    # remove axis labels
    cg.ax_heatmap.set_ylabel(None)
    cg.ax_heatmap.set_xlabel(None)

    # add spines
    for _, spine in cg.ax_heatmap.spines.items():
        spine.set_visible(True)
        spine.set_lw(spinewidth)
        
    x_le,y_le = label_every
    x_at,y_at = alternate_ticks
    x_atl,y_atl = alternate_tick_length
    ytickrotation = yticklabel_kwargs.pop('rotation','horizontal')
    xtickrotation = xticklabel_kwargs.pop('rotation','vertical')
    # set up tick labels, enabling alternating offsets
    cg.ax_heatmap.xaxis.set_major_locator(FixedLocator(np.linspace(0.5,x_len-(x_len-0.5)%(x_le*2),int(np.ceil((x_len-0.5)/(2*x_le))))))
    cg.ax_heatmap.xaxis.set_minor_locator(FixedLocator(np.linspace(0.5+x_le,x_len-(x_len-0.5-x_le)%(x_le*2),int(np.ceil((x_len-0.5-x_le)/(2*x_le))))))
    cg.ax_heatmap.yaxis.set_major_locator(FixedLocator(np.linspace(0.5,y_len-(y_len-0.5)%(y_le*2),int(np.ceil((y_len-0.5)/(2*y_le))))))
    cg.ax_heatmap.yaxis.set_minor_locator(FixedLocator(np.linspace(0.5+y_le,y_len-(y_len-0.5-y_le)%(y_le*2),int(np.ceil((y_len-0.5-y_le)/(2*y_le))))))
    _ = cg.ax_heatmap.set_yticklabels(df.index.get_level_values(0)[::2*y_le],fontsize=label_fontsize,rotation=ytickrotation,**yticklabel_kwargs)
    _ = cg.ax_heatmap.set_yticklabels(df.index.get_level_values(0)[y_le::2*y_le],minor=True,fontsize=label_fontsize,rotation=ytickrotation,**yticklabel_kwargs)
    _ = cg.ax_heatmap.set_xticklabels(df.columns.get_level_values(0)[::2*x_le],fontsize=label_fontsize,rotation=xtickrotation,**xticklabel_kwargs)
    _ = cg.ax_heatmap.set_xticklabels(df.columns.get_level_values(0)[x_le::2*x_le],minor=True,fontsize=label_fontsize,rotation=xtickrotation,**xticklabel_kwargs)

    if y_at:
        cg.ax_heatmap.tick_params(axis='y',which='major',pad=2,length=2)
        cg.ax_heatmap.tick_params(axis='y',which='minor',pad=2,length=y_atl)
    else:
        cg.ax_heatmap.tick_params(axis='y',which='both',pad=2,length=2)
    if x_at:
        cg.ax_heatmap.tick_params(axis='x',which='major',pad=2,length=2)
        cg.ax_heatmap.tick_params(axis='x',which='minor',pad=2,length=x_atl)
    else:
        cg.ax_heatmap.tick_params(axis='x',which='both',pad=2,length=2)

    return cg

def boxplot_jitter(jitter_ax="x", jitter_range=(-0.25, 0.25), *args, **kwargs):

    kwargs.setdefault('flierprops',
        dict(marker=".", markeredgecolor="none", alpha=0.1, markersize=5)
    )
    ax = sns.boxplot(*args, **kwargs)

    if jitter_ax not in ["x", "y"]:
        raise ValueError(f'`jitter_ax` must be one of {"x","y"}')
    else:
        ax_ = int(jitter_ax == "y")

    for line in ax.get_lines()[5::6]:
        data = np.array(line.get_data())
        data[ax_] = data[ax_] + np.random.uniform(*jitter_range, data[ax_].size)
        line.set_data(data)

    return ax

def symlog_axis(vals,ax,which):
    if which=='x':
        ax.set_xscale("symlog", linthresh=1, basex=10, subs=np.arange(1, 11))
        op_ax = ax.xaxis
    elif which=='y':
        ax.set_yscale("symlog", linthresh=1, basex=10, subs=np.arange(1, 11))
        op_ax = ax.yaxis
    else:
        raise ValueError(f'which must be one of "x" or "y".')

    op_ax.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=0.1, subs=np.arange(1, 10))
    )
    ax_min_sign, ax_max_sign = np.sign(vals.min()), np.sign(vals.max())
    ax_min, ax_max = np.ceil(np.log10(abs(vals.min()))), np.ceil(
        np.log10(abs(vals.max()))
    )
    op_ax.set_view_interval(ax_min_sign * 10 ** ax_min, ax_max_sign * 10 ** ax_max)

    ticklabels = []
    ticks = []
    if ax_min_sign == -1:
        if ax_max_sign == -1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_min - ax_max) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, ax_max, int(ax_min - ax_max) + 1))
        else:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, 0, int(ax_min) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, 0, int(ax_min) + 1))

    if ax_max_sign == 1:
        if ax_min_sign == 1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1)
                ]
            )
            ticks.append(np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1))
        else:
            ticklabels.append("0")
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(0, ax_max, int(ax_max) + 1)
                ]
            )
            ticks.append(np.array([0]))
            ticks.append(np.logspace(0, ax_max, int(ax_max) + 1))

    op_ax.set_ticks(np.concatenate(ticks))
    op_ax.set_ticklabels(ticklabels)
    return ax


def get_cp_feature_table(
    compartments=["cell", "nucleus"],
    channels=["dapi", "tubulin", "gh2ax", "phalloidin"],
    foci_channel="gh2ax",
    neighbor_distances=[1],
):

    from ops.cp_emulator import (
        intensity_columns_multichannel,
        intensity_distribution_columns_multichannel,
        texture_columns_multichannel,
        correlation_columns_multichannel,
        texture_columns_multichannel,
        shape_columns,
        shape_features as shape_features_,
    )

    intensity_features = []
    for v in intensity_columns_multichannel.values():
        intensity_features.extend(v)

    distribution_features = []
    for v in intensity_distribution_columns_multichannel.values():
        distribution_features.extend(v)

    texture_features = []
    for v in texture_columns_multichannel.values():
        texture_features.extend(v)

    correlation_features = []
    for v in correlation_columns_multichannel.values():
        correlation_features.extend(v)

    shape_features = list(shape_features_.keys())

    _ = [
        shape_features.remove(f)
        for f in ["zernike", "centroid", "feret_diameter", "radius", "hu_moments"]
    ]

    shape_features.extend(list(shape_columns.values()))

    shape_features.extend([f"hu_moments_{n}" for n in range(7)])

    if foci_channel is not None:
        from ops.features import foci

        df_foci = (
            pd.DataFrame(foci.keys(), columns=["feature_type"])
            .assign(level_0="cell")
            .assign(level_1=foci_channel)
            .assign(level_2="intensity")
        )
    else:
        df_foci = None

    df_grayscale = pd.concat(
        [df_foci]
        + [
            (
                pd.DataFrame(intensity_features, columns=["feature_type"]).assign(
                    level_0=compartment, level_1=channel, level_2="intensity"
                )
            )
            for compartment in compartments
            for channel in channels
        ]
        + [
            (
                pd.DataFrame(distribution_features, columns=["feature_type"]).assign(
                    level_0=compartment, level_1=channel, level_2="distribution"
                )
            )
            for compartment in compartments
            for channel in channels
        ]
        + [
            (
                pd.DataFrame(texture_features, columns=["feature_type"]).assign(
                    level_0=compartment, level_1=channel, level_2="texture"
                )
            )
            for compartment in compartments
            for channel in channels
        ]
    ).assign(
        feature=lambda x: (
            x.apply(
                lambda x: x[["level_0", "level_1", "feature_type"]].str.cat(sep="_"),
                axis=1,
            )
        )
    )

    df_correlation = (
        pd.DataFrame(
            [
                {
                    "feature_type": feature.rsplit("_", 2)[0],
                    "level_0": compartment,
                    "level_1": "correlation",
                    "level_2": (f"{first}_{second}"),
                }
                for feature in correlation_features
                for compartment in compartments
                for first, second in combinations(channels, r=2)
            ]
        )
        .drop_duplicates()
        .assign(
            feature=lambda x: (
                x.apply(
                    lambda x: x[["level_0", "feature_type", "level_2"]].str.cat(
                        sep="_"
                    ),
                    axis=1,
                )
            )
        )
    )

    df_correlation = pd.concat(
        [
            df_correlation,
            (
                df_correlation.query(
                    'feature_type in ["K","manders","rwc","lstsq_slope"]'
                ).assign(
                    feature=lambda x: (
                        x["feature"].apply(
                            lambda x: "_".join(
                                x.rsplit("_", 2)[:1] + x.rsplit("_", 2)[-1:0:-1]
                            )
                        )
                    )
                )
            ),
        ]
    )

    df_shape = pd.concat(
        [
            (
                pd.DataFrame(shape_features, columns=["feature_type"]).assign(
                    level_0=compartment, level_1="shape", level_2="shape"
                )
            )
            for compartment in compartments
        ]
    ).assign(
        feature=lambda x: (
            x.apply(lambda x: x[["level_0", "feature_type"]].str.cat(sep="_"), axis=1)
        )
    )

    df_neighbors = pd.concat(
        [
            (
                pd.DataFrame(
                    [
                        {
                            "feature_type": f"{feature}_{neighbor_distance}",
                            "level_0": compartment,
                            "level_1": "neighbors",
                            "level_2": "neighbors",
                        }
                        for feature in ["number_neighbors", "percent_touching"]
                        for compartment in compartments
                        for neighbor_distance in neighbor_distances
                    ]
                )
            ),
            (
                pd.DataFrame(
                    [
                        {
                            "feature_type": feature,
                            "level_0": compartment,
                            "level_1": "neighbors",
                            "level_2": "neighbors",
                        }
                        for feature in [
                            "first_neighbor_distance",
                            "second_neighbor_distance",
                            "angle_between_neighbors",
                        ]
                        for compartment in compartments
                    ]
                )
            ),
        ]
    ).assign(
        feature=lambda x: (
            x.apply(lambda x: x[["level_0", "feature_type"]].str.cat(sep="_"), axis=1)
        )
    )

    return pd.concat([df_grayscale, df_correlation, df_shape, df_neighbors]).set_index(
        "feature"
    )
