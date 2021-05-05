import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import pandas as pd

def volcano(
	    df, feature=None, x='score', y='pval', alpha=0.05, change=0, annotate=None, 
	    prelog=True, xscale=None, control_query=None, control_label='non-targeting guides', 
	    ax=None
	    ):

	df_ = df.copy()

	if ax is None:
		fig,ax = plt.subplots(1,1,figsize=(7,7))

	# `feature` is not None, `x` and `y` are suffices for paired columns
	if feature is not None:
		x = f'{feature}_{x}'.strip('_')
		y = f'{feature}_{y}'.strip('_')

	# `alpha` is None, nothing is marked as significant
	if alpha is None:
		alpha = df_[y].min()-0.1

	# `change` is a single value, make cutoffs symmetric around 0
	if not isinstance(change,list):
		try:
			change = [-change,change]

		# negative fails for None
		except:
			change = [change,change]

	# None in `change` -> nothing marked marked as significant in that direction
	# if both elements of `change` are None, everything marked as significant
	if change==[None,None]:
		change = [df_[x].min()-1,]*2
	else:
		if change[0] is not None:
			ax.axvline(change[0],color='gray',linestyle='--')
		else:
			change[0] = df_[x].min()-1

		if change[1] is not None:
			ax.axvline(change[1],color='gray',linestyle='--')
		else:
			change[1] = df_[x].max()+1

	df_['significant'] = False
	df_.loc[(df_[x]<change[0])&(df_[y]<alpha),'significant'] = 'low'
	df_.loc[(df_[x]>change[1])&(df_[y]<alpha),'significant'] = 'high'

	if prelog:
		df_[y] = -np.log10(df_[y])
		alpha = -np.log10(alpha)

	if control_query is not None:
		df_control = df_.query(control_query)
		df_ = df_[~(df_.index.isin(df_control.index))]

	sns.scatterplot(
		data=df_, x=x, y=y, hue='significant', hue_order=['high',False,'low'],
		palette=['green','gray','magenta'], legend=None, ax=ax
		)

	if control_query is not None:
		sns.scatterplot(
			data=df_control, x=x, y=y, color=sns.color_palette()[1], ax=ax,
			label=control_label)

	# sns.scatterplot(data=df_.query('gene_symbol==@annotate_labels'),x=x,
	#                 y=y,hue='significant',hue_order=['high','low'],palette=['green','magenta'],edgecolor='black',ax=ax)

	if not prelog:
		ax.set_yscale('log',basey=10)
		y_max = np.floor(np.log10(df_[y].min()))
		ax.set_ylim([1.15,10**(y_max)])
		ax.set_yticks(np.logspace(y_max,0,-(int(y_max)-1)))
		ax.set_yticklabels(
			labels=[
			f'$\\mathdefault{{10^{{{int(n)}}}}}$' for n in np.linspace(y_max,0,-(int(y_max)-1))
			],
			fontsize=14
			)

	if xscale=='symlog':
		ax.set_xscale('symlog',linthresh=1,basex=10,subs=np.arange(1,11))
		ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10,linthresh=0.1,subs=np.arange(1,10)))
		x_min_sign,x_max_sign = np.sign(df_[x].min()),np.sign(df_[x].max())
		x_min,x_max = np.ceil(np.log10(abs(df_[x].min()))),np.ceil(np.log10(abs(df_[x].max())))
		ax.set_xlim([x_min_sign*10**x_min,x_max_sign*10**x_max])

		xticklabels = []
		xticks = []
		if x_min_sign==-1:
			if x_max_sign==-1:
				xticklabels.extend([
					f'$\\mathdefault{{-10^{{{int(n)}}}}}$' for n in np.linspace(x_min,x_max,int(x_min-x_max)+1)
					])
				xticks.append(-np.logspace(x_min,x_max,int(x_min-x_max)+1))
			else:
				xticklabels.extend([
					f'$\\mathdefault{{-10^{{{int(n)}}}}}$' for n in np.linspace(x_min,0,int(x_min)+1)
					])
				xticks.append(-np.logspace(x_min,0,int(x_min)+1))
		
		if x_max_sign==1:
			if x_min_sign==1:
				xticklabels.extend([
					f'$\\mathdefault{{10^{{{int(n)}}}}}$' for n in np.linspace(x_min,x_max,int(x_max-x_min)+1)
					])
				xticks.append(np.linspace(x_min,x_max,int(x_max-x_min)+1))
			else:
				xticklabels.append('0')
				xticklabels.extend([f'$\\mathdefault{{10^{{{int(n)}}}}}$' for n in np.linspace(0,x_max,int(x_max)+1)])
				xticks.append(np.array([0]))
				xticks.append(np.logspace(0,x_max,int(x_max)+1))

		ax.set_xticks(np.concatenate(xticks))
		ax.set_xticklabels(labels=xticklabels,fontsize=14)

	# for y_offset,alignment,label in annotate:
	#     s = df_.query('gene_symbol==@label').iloc[0]
	#     ax.text(s[x],s[y]+y_offset,
	#             label,horizontalalignment=alignment,fontsize=12)
	
	ax.axhline(alpha,color='gray',linestyle='--')

	ax.set_xlabel(' '.join(x.split('_')),fontsize=14)
	if prelog:
		ax.set_ylabel('-log10(p-value)',fontsize=14)
	else:
		ax.set_ylabel('p-value',fontsize=14)
	
	return ax