#!/usr/bin/env python3

# Permuting per-event PS-accuracy - dependency relationship
# How null is our null?

Nshuff = 10

import tqdm as tqdm
import pandas as pd
import numpy as np
import deepdish as dd

# High/low performing subjects:
dependencydf = pd.read_csv('csvs/Dependency_Year_1.csv')
excludedf = dependencydf[(dependencydf['Accuracy']<0.3) | (dependencydf['Accuracy']>0.95)]
exclude_subjs = excludedf['Subject']

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
# Suppress convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)

zvals = {k:{k:{k:[] for k in ['Full','Smaller']} for k in ['All data','No perfectly dependent events','No good/bad subjects']} for k in [True,False]}
for delay in [True,False]:
	tmp = pd.read_csv('csvs/All participants Delay = '+str(delay)+'.csv')
	print(delay)
	# permutation needs to be done on all subjects
	# switch the Targets for all 18 events
	# THEN eliminate events with perfect dependency (0 or 6)
	for i in tqdm.tqdm(range(Nshuff+1)):
		# Set a new random seed for each iteration
		np.random.seed(i)
		# Step 1: Group the DataFrame by 'Subject'
		groups = {subject: group for subject, group in tmp.groupby('Subject')}
		# Step 2: Shuffle the order of the Subject keys
		shuffled_subjects = np.random.permutation(list(groups.keys()))
		# Step 3: Reassemble the DataFrame with shuffled subjects
		shuffled_df = pd.concat([groups[subject] for subject in shuffled_subjects], ignore_index=True)
		tmprand =pd.concat([shuffled_df[['Subject','Age','Sex','Event','Foil','Target','Lure']],  tmp[['Dependency','N PC correct','group']]], axis=1)
		if i == 0: # No shuffling
			tmprand =pd.concat([tmp[['Subject','Age','Sex','Event','Foil','Target','Lure']],  tmp[['Dependency','N PC correct','group']]], axis=1)
		# Exclude perfectly dependent events:
		tmprand_ = tmprand[(tmprand['N PC correct'] != 0) & (tmprand['N PC correct'] != 6)]
		tmprand_ex = tmprand_[~tmprand_.Subject.isin(exclude_subjs)]
		for k,d in {'All data':tmprand,'No perfectly dependent events':tmprand_,'No good/bad subjects':tmprand_}.items():
			vcf = {'Event': '0 + C(Event)', 'Subject': '0 + C(Subject)'}
			model2 = sm.MixedLM.from_formula("Dependency ~ Target", groups="group", vc_formula=vcf, re_formula="0",data=d)
			result2 = model2.fit(method="cg")
			zvals[delay][k]['Full'].append(result2.tvalues["Target"])
			model1 = smf.mixedlm("Dependency ~ Target", d, groups=d["Subject"]) # Random Intercept
			result1 = model1.fit()
			zvals[delay][k]['Smaller'].append(result1.tvalues["Target"])
			if i==0:
				print(delay,k)
				print(result1.summary())

dd.io.save('csvs/Permuted_events.h5',zvals)
zvals = dd.io.load('csvs/Permuted_events.h5')
			
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid",font_scale=1.5)
for delay in [True,False]:
	for k in ['All data','No perfectly dependent events','No good/bad subjects']:
		for model in ['Full','Smaller']:
			df = pd.DataFrame({'Z-values':zvals[delay][k][model][1:]})
			fig, axs = plt.subplots(1, figsize=(4, 4))
			sns.violinplot(y=df['Z-values'],ax=axs,color="grey",linewidth=0)
			axs.plot(0,zvals[delay][k][model][0],'k*',markersize=14)
			for z in [-1.96,1.96]:
				axs.axhline(y=z, color='k', linestyle='--',alpha=0.5,linewidth=4)
			axs.set_title('Delay: '+str(delay)+'\n'+k+'\n'+model+' model')
