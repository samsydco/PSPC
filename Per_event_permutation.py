#!/usr/bin/env python3

# Permuting per-event PS-accuracy - dependency relationship
# How null is our null?

Nshuff = 100
year = 1

import tqdm as tqdm
import pandas as pd
import numpy as np
import deepdish as dd

# High/low performing subjects AND pilot subjects:
dependencydf = pd.read_csv('csvs/Dependency_Year_1.csv')
excludedf = dependencydf[(dependencydf['Accuracy']<0.3) | (dependencydf['Accuracy']>0.95) | (dependencydf['Same Day'] == False)]
exclude_subjs = excludedf['Subject']

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
# Suppress convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)

vals = {k:{k:{k:{k:{k:[] for k in ['Full','Smaller']} for k in ['Target','Lure','Foil']} for k in ['All data','No perfectly dependent events','No perfect subjects']} for k in ['zval','pval']} for k in ['Dependency','N_PC_correct']}

tmp = pd.read_csv('csvs/Per_event_Year_'+str(year)+'.csv')
tmp = tmp.rename(columns={'N PC correct': 'N_PC_correct'})
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
	tmprand =pd.concat([shuffled_df[['Subject','Age','Sex','Event','Foil','Target','Lure','Delay']],  tmp[['Dependency','N_PC_correct','group']]], axis=1)
	if i == 0: # No shuffling
		tmprand =pd.concat([tmp[['Subject','Age','Sex','Event','Foil','Target','Lure','Delay']],  tmp[['Dependency','N_PC_correct','group']]], axis=1)
	# Exclude perfectly dependent events:
	tmprand_ = tmprand[(tmprand['N_PC_correct'] != 0) & (tmprand['N_PC_correct'] != 6)]
	tmprand_ex = tmprand_[~tmprand_.Subject.isin(exclude_subjs)]
	for selection in ['Target','Lure','Foil']:
		#d_ = {'All data':tmprand,'No perfectly dependent events':tmprand_,'No perfect subjects':tmprand_}
		d_ = {'No perfect subjects':tmprand_}
		for k,d in d_.items():
			for pc in ['Dependency','N_PC_correct']:
				vcf = {'Event': '0 + C(Event)', 'Subject': '0 + C(Subject)'}
				model2 = sm.MixedLM.from_formula(pc+" ~ "+selection+" + Delay", groups="group", vc_formula=vcf, re_formula="0",data=d)
				result2 = model2.fit(method="cg")
				vals[pc]['zval'][k][selection]['Full'].append(result2.tvalues[selection])
				vals[pc]['pval'][k][selection]['Full'].append(result2.pvalues[selection])
				model1 = smf.mixedlm(pc+" ~ "+selection+" + Delay", d, groups=d["Subject"]) # Random Intercept
				result1 = model1.fit()
				vals[pc]['zval'][k][selection]['Smaller'].append(result1.tvalues[selection])
				vals[pc]['pval'][k][selection]['Smaller'].append(result1.pvalues[selection])
				if i==0:
					print(selection,k,pc)
					#print('z-value: ',result1.tvalues[selection])
					#print('p-value: ',result1.pvalues[selection])
					# r-value from t-value:
					t_value = result1.tvalues[selection]
					df = result1.df_resid
					r_value = np.sqrt(np.square(t_value) / (np.square(t_value) + df))
					print(result1.summary())

dd.io.save('csvs/Permuted_events.h5',vals)
vals = dd.io.load('csvs/Permuted_events.h5')
			
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid",font_scale=1.5)
order = ['Target (Correct)','Lure (Similar)','Foil']
for pc in ['Dependency','N_PC_correct']:
	print(pc)
	for k in ['No perfect subjects']:#['All data','No perfectly dependent events','No perfect subjects']:
		for model in ['Smaller']:
			fig, axs = plt.subplots(1,3, figsize=(12, 4))
			for i,selection in enumerate(['Target','Lure','Foil']):
				df = pd.DataFrame({'Z-values':vals[pc]['zval'][k][selection][model][1:]})

				sns.violinplot(y=df['Z-values'],ax=axs[i],color="grey",linewidth=0)
				axs[i].plot(0,vals[pc]['zval'][k][selection][model][0],'k*',markersize=14)
				for z in [-1.96,1.96]:
					axs[i].axhline(y=z, color='k', linestyle='--',alpha=0.5,linewidth=4)
				axs[i].set_title(order[i])#(k+'\n'+order[i]+'\n'+'p = '+str(np.round(vals['pval'][k][selection][model][0],2)))
			plt.tight_layout()
			if pc == 'Dependency':
				plt.savefig('Figures/Figure4.tif', dpi=300,bbox_inches="tight",format='tif')
			plt.show()
