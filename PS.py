#!/usr/bin/env python3

# Analyse PS
# break out targets, lures, and foils

import tqdm as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepdish as dd
import seaborn as sns
import math
from scipy import stats

year = 1

figurepath = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
datadf = pd.read_csv('datadf.csv')
Contdict = dd.io.load('PSPC_cont_tables.h5')
datadf = datadf[datadf['Year'] == year]


PScols = ['object','location','animal']
PSkey = {-1:'Foil',1:'Target',0:'Lure'}

output = []
percatoutput = []
for subject,tmp in tqdm.tqdm(Contdict[year].items()):
	PStmp = tmp[PScols].to_numpy()
	tmpdf = datadf[datadf['Subject']==subject]
	age = tmpdf['Age'].values[0]
	delay=tmpdf['Delay'].iloc[0]
	nitems = np.count_nonzero(~np.isnan(PStmp.astype(float))) # PStmp.size (if did both blocks)
	d_ = {'Subject': subject,'Delay':delay,'Age':age}
	d = d_.copy()
	for col in PScols:
		d[col+' target selection rate'] = np.count_nonzero(tmp[col] == 1) / (nitems/3)
	percatoutput.append(d)
	for value in PSkey.keys():
		d2 = d_.copy()
		d2['Selection'] = PSkey[value]
		d2['Proportion Selected']  = np.count_nonzero(PStmp == value) / nitems
		output.append(d2)
	
outputdf = pd.DataFrame(output)
percatoutputdf = pd.DataFrame(percatoutput)
outputdf.to_csv('PS_Year_'+str(year)+'.csv',index=False)
percatoutputdf.to_csv('PS_cat_Year_'+str(year)+'.csv',index=False)
outputdf['Age'] = outputdf['Age'].map(lambda age: math.floor(age))
delaydf = outputdf[(outputdf.Delay)]
nodelaydf = outputdf[(outputdf.Delay == False)]

order = ['Target','Lure','Foil']
orderage = [4,5,6,7]


sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=outputdf, x="Selection", y="Proportion Selected", hue="Age", palette="vlag",order=order, showfliers = False)
sns.stripplot(data=outputdf, x="Selection", y="Proportion Selected", hue="Age", dodge=True,color=".3",order=order, jitter=False)
vs = {k:[] for k in orderage}
for i,points in enumerate(ax.collections):
	vertices = points.get_offsets().data
	if len(vertices)>0:
		for ii,a in enumerate([4,7,6,5]):
			if (i+ii) % 4 == 0:
				vs[a].append(vertices[0][0])
for subject in delaydf.Subject.unique():
	tmp = delaydf[delaydf.Subject == subject]
	age = tmp['Age'].iloc[0]
	for i,selection in enumerate(order):
		data = tmp[tmp.Selection==selection]['Proportion Selected'].iloc[0]
		ax.scatter(vs[age][i],data,marker='*',color='r',alpha=0.75, s=300,zorder=3)

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel=None)
plt.axhline(y=0.25, color='k', linestyle='--')
fig.savefig(figurepath+'PS_Target_Lure_Foil.png', bbox_inches='tight', dpi=100)


# plot correlation between age and Target/Lure/Foil proportion 
outputdf = pd.DataFrame(output)
sns.set_theme(style="white", palette=['#6e90bf',  '#d9a6a4', '#c26f6d'])
g = sns.lmplot(
    data=outputdf,
    x="Age", y="Proportion Selected", hue="Selection",
)
plt.axhline(y=0.25, color='k', linestyle='--')
g.savefig(figurepath+'PS_corr_Target_Lure_Foil.png', bbox_inches='tight', dpi=100)
statlist = []
for selection in PSkey.values():
	tmpdf = outputdf[outputdf['Selection'] == selection]
	res = stats.pearsonr(tmpdf['Age'],tmpdf['Proportion Selected'])
	statlist.append({'Selection':selection, 'Age/Prop r':res[0], 'Age/Prop p':res[1]})
statdf=pd.DataFrame(statlist)
statdf

# cannot run rmANOVA due to unequal group sizes (unequal number of subjects in each age group)
# Therefore running LMM
import statsmodels.formula.api as smf

outputdf['Age'] = outputdf['Age'].map(lambda age: math.floor(age))
outputdf = pd.DataFrame(output)
outputdf.rename(columns={'Proportion Selected': 'PS'}, inplace=True)
outputdf['Selection'] = outputdf['Selection'].replace(['Foil','Target','Lure'], [-1,1,0])
md = smf.mixedlm('PS~Age+Selection+Age*Selection',outputdf,groups=outputdf["Subject"])
mdf = md.fit()
print('Model with all possible items: target/lure/foil:')
print(mdf.summary())
for k,v in PSkey.items():
	outputdf = pd.DataFrame(output)
	outputdf.rename(columns={'Proportion Selected': 'PS'}, inplace=True)
	outputdf['Selection'] = outputdf['Selection'].replace(['Foil','Target','Lure'], [-1,1,0])
	outputdf = outputdf.drop(outputdf[outputdf.Selection == k].index)
	md = smf.mixedlm('PS~Age+Selection+Age*Selection',outputdf,groups=outputdf["Subject"])
	mdf = md.fit()
	print('Model without '+v)
	print(mdf.summary())

import pingouin as pg
outputdf = pd.DataFrame(output)
res = pg.rm_anova(dv='Proportion Selected',within='Selection',subject='Subject',data=outputdf)
post_hocs = pg.pairwise_tests(dv='Proportion Selected',within='Selection',subject='Subject',padjust='bonf',data=outputdf)


	
	


