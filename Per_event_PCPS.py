#!/usr/bin/env python3

# Associate memory coherence (PC) to memory discrimination (PS) on an event-by-event level

# Memory Coherence:
# 1) Get joint-retrieval of each pair in each event (ABAC, BABC, ...)
# --> 1 if equal (both 0 or both 1)
# --> 0 if unequal (one 1 and one 0)
# --> Sum that up and average across 6 pairs
# 2) Calculate independent model for each of 6 pairs (averaged across events)
# 3) For all 6 pairs: Subtract joint-retrieval by independent model --> Average dependency for each event (averaged across all 6 pairs)
# 4) Get event-specific memory discrimination: Did they get animal, place, and object correct?
# 5) Linear mixed model: Does event dependency related to PS?
# 6) Exclude ceiling and floor coherence trials and run again

import tqdm as tqdm
import deepdish as dd
import pandas as pd
import numpy as np


year = 1

datadf = pd.read_csv('csvs/datadf.csv')
Contdict = dd.io.load('csvs/PSPC_cont_tables.h5')
datadf = datadf[datadf['Year'] == year]

PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]

PScols = ['object','location','animal']
PSkey = {-1:'Foil',1:'Target',0:'Lure'}

dflist = []
for subject,res_tmp in tqdm.tqdm(Contdict[year].items()):
	if np.count_nonzero(~np.isnan(res_tmp[PCcols].astype(float))) < 108:
		print(subject+' Did not complete both blocks of experiment')
	else:
		tmp = datadf[datadf['Subject']==subject]
		age = tmp['Age'].values[0]
		sex = tmp['Gender'].iloc[0]
		delay = tmp['Delay'].values[0]
		d = {'Subject': subject,'Delay':delay,'Age':age,'Sex':sex}
		joint_retrieval = np.zeros((len(pair_array),len(res_tmp)))
		dep = np.zeros((len(pair_array),len(res_tmp)))
		indi_model = []
		for i,pair in enumerate(pair_array):
			res2 = res_tmp[pair]
			joint_retrieval[i] = list(res2.sum(axis=1)!=1)
			acc = res2.mean()
			indi_model.append((acc[0]*acc[1])+((1-acc[0])*(1-acc[1])))
			dep[i] = joint_retrieval[i] - indi_model[i]
		event_joint_retrieval = np.mean(joint_retrieval,axis=0)
		avg_dep = np.mean(dep,axis=0)
		# Pattern Separation:
		PStmp = res_tmp[PScols].to_numpy()
		mem_disc = np.zeros((len(PSkey),len(PStmp)))
		for i,v in enumerate(PSkey.keys()):
			mem_disc[i] = np.mean(PStmp == v,axis=1)
		for i in range(len(avg_dep)):
			d2 = d.copy()
			d2['Event'] = i
			d2['Dependency'] = avg_dep[i]
			d2['N PC correct'] = res_tmp[PCcols].sum(axis=1)[i]
			for ii,k in enumerate(PSkey.values()):
				d2[k] = mem_disc[ii,i]
			dflist.append(d2)

df = pd.DataFrame(dflist)
df["group"] = 1
# Exclude perfectly dependent events:
df_ = df[(df['N PC correct'] != 0) & (df['N PC correct'] != 6)]
# OR Exclude perfectly dependent subjects:
dependencydf = pd.read_csv('csvs/Dependency_Year_1.csv')
excludedf = dependencydf[dependencydf['Accuracy']>0.95]
exclude_subjs = excludedf['Subject']
df_ex = df_[~df_.Subject.isin(exclude_subjs)]


import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy import stats
# Suppress convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)
for kd,d in {'All participants':df,'Removing events with perfect dependency':df_,'Removing events and subjects with perfect dependency':df_ex}.items():
	tmp = d
	for k in ['Target','Lure','Foil','N PC correct']:#PSkey.values():
		if k == 'N PC correct':
			tmp = tmp.rename(columns={'N PC correct': 'N_PC_correct'})
			k = 'N_PC_correct'
		# Following this advice: 
		# https://stackoverflow.com/questions/50052421/mixed-models-with-two-random-effects-statsmodels?rq=3
		vcf = {'Event': '0 + C(Event)', 'Subject': '0 + C(Subject)'}
		model2 = sm.MixedLM.from_formula("Dependency ~ "+k+" + Delay", groups="group", vc_formula=vcf, re_formula="0",data=tmp)
		result2 = model2.fit(method="cg")
		if result2.pvalues[k] != 0.00:
			print('Model for '+k)
			print(kd)
			print('With both Event and Subject (statsmodels implementation):')
			print(result2.summary())
			model1 = smf.mixedlm("Dependency ~ "+k+" + Delay", tmp, groups=tmp["Subject"]) # Random Intercept
			result1 = model1.fit()
			print('Model for '+k)
			print(kd)
			print('With only Subject (statsmodels implementation):')
			print(result1.summary())
				

			
			
from statsmodels.formula.api import ols
model = ols("Dependency ~ Target + Subject + Event", data=df[df['Delay']==delay]).fit()
print(model.summary())


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Aidan's question: Does dependnecy averaged across all 18 events equal dependency when calculated the "normal" way?
df = df.join(
    df.groupby('Subject')['Dependency'].transform('mean').rename('Dependency Averaged across all events')  # Rename columns 
)
dftmp = df[df['Event'] == 0]
newdf = pd.merge(dependencydf, dftmp, on="Subject")
newdf = newdf.rename({'Dependency_x':'Dependency'},axis=1)
stat_vals = stats.pearsonr(newdf['Dependency'],newdf['Dependency Averaged across all events'])
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.regplot(data=newdf, x="Dependency", y='Dependency Averaged across all events', ax=ax)

# Draw a scatter plot while assigning point colors and sizes to different
# variables in the dataset
tmp=df_ex
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Target", y="Dependency",
				#hue="Event", size="Subject",
                hue="Subject", size="Event",#sizes=(1, 18), 
				palette="ch:r=-.2,d=.3_r",
                legend=False,#linewidth=0,
                data=tmp[tmp['Delay']==delay], ax=ax, alpha=0.5)
sns.regplot(data=tmp[tmp['Delay']==delay], x="Target", y="Dependency",ax=ax,scatter=False)
ax.set_xlabel('Proportion of Target selected per event')
ax.set_ylabel('Event-level dependency')
	
# Distribution plots:
bins=10
f,ax = plt.subplots(1,2)
for i,delay in enumerate([True,False]):
	# Distrubiton of dependency:
	ax[i].hist([list(df[df['Delay']==delay]['Dependency']),list(df_[df_['Delay']==delay]['Dependency'])],
			   bins,alpha=0.5,label=['All participants','Removing events with perfect dependency'])
	ax[i].set_title('Delay = '+str(delay))
	ax[i].set_xlabel('Per event Dependency')
ax[0].set_ylabel("Count")
ax[1].legend(bbox_to_anchor=(1.1, 1.05))

delaylabs = ['Immediate','Delayed']
def per_event_histogram(df,column,connect_dots):	
	f,ax = plt.subplots(1,2, figsize=(10, 5))
	for i,delay in enumerate([False,True]):
		tmp = df[df['Delay']==delay]
		# Distrubiton of dependency:
		g = sns.boxplot(
			tmp, x="Event", y=column, hue="Event",whis=[0, 100],
			width=.6,palette="vlag",ax=ax[i],legend=False
		)
		# Add in points to show each observation
		if connect_dots == False:
			sns.stripplot(tmp, x="Event", y=column, size=3, ax=ax[i],color=".3")#,hue='Subject',legend=False)#,color=".3")
		else:
			for event in range(18):
				vals = tmp[tmp['Event'] == event][column]
				ax[i].plot(event*np.ones((len(vals),1)), vals, 'o', alpha=.40, zorder=1, ms=6, mew=1)
			for subj in tmp['Subject'].unique():
				ax[i].plot(tmp[tmp['Subject'] == subj]['Event'], tmp[tmp['Subject'] == subj][column], color = 'grey', linewidth = 0.5, linestyle = '-', zorder=-1)
		
		ax[i].axhline(y=0.0, color='k', linestyle='--')
		ax[i].set_title(delaylabs[i])
		if column=='Dependency':
			ax[i].set_ylim([-0.85,0.55])
		g.set(xticklabels=[])
	g.set(yticklabels=[])
	g.set(ylabel=None)

for column in ['Dependency']:#,'N PC correct','Target','Lure','Foil']:
	for k,v in {'All events':df,'Excluding events with perfect dependency':df_,'Excluding dependent subjects and events with perfect dependency':df_ex}.items():
		print(k,column)
		per_event_histogram(v,column,True)
	
# Per subject histogram:
def new_col(row):
	subj = row['Subject']
	return dependencydf[dependencydf['Subject']==row['Subject']]['Accuracy'].iloc[0]

def per_subject_histogram(df,column):
	f,ax = plt.subplots(2,1, figsize=(22, 7))
	for i,delay in enumerate([False,True]):
		tmp = df[df['Delay']==delay].copy()
		tmp["Accuracy"] = df.apply(new_col, axis=1)
		tmp = tmp.sort_values(by=['Accuracy','Subject']).reset_index(drop=True)
		tmp['Accuracy'] = tmp['Accuracy'].round(2)
		# Distrubiton of dependency:
		g = sns.boxplot(
			tmp, x="Subject", y=column, hue="Subject",whis=[0, 100],
			width=.6,palette="vlag",ax=ax[i],legend=False
		)
		# Add in points to show each observation
		sns.stripplot(tmp, x="Subject", y=column, size=3, color=".3",ax=ax[i])
		ax[i].axhline(y=0.0, color='k', linestyle='--')
		ax[i].set_title(delaylabs[i])
		if column == 'Dependency':
			ax[i].set_ylim([-0.85,0.55])
		accvals = np.round(np.sort(dependencydf[dependencydf['Subject'].isin(list(tmp['Subject']))]['Accuracy']),2)
		g.set_xticks(ax[i].get_xticks())
		g.set_xticklabels(list(accvals), rotation=45)
		g.set(xlabel='Accuracy')
	plt.tight_layout()

sns.set(style="whitegrid", font_scale=1.5)
for column in ['Dependency']:#,'N PC correct','Target','Lure','Foil']:
	for k,v in {'All events':df,'Excluding events with perfect dependency':df_,'Excluding dependent subjects and events with perfect dependency':df_ex}.items():
		print(k,column)
		per_subject_histogram(v,column)