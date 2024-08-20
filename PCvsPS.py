#!/usr/bin/env python3

# PC vs PS
# Each subject gets one point (average PC and average PS across all trials)

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as sm2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="white",font_scale=1.5, palette=['#0D096E','#890505'])
sns.set_theme(style="white",font_scale=2, palette=['#0D096E','#890505'])

PSdf = pd.read_csv('csvs/PS_cat_Year_1.csv')
PCdf = pd.read_csv('csvs/Dependency_Year_1.csv')
combodf = pd.merge(PCdf, PSdf, how='outer', indicator=True)
missingdf = combodf[combodf['_merge'].str.contains('only')]
if len(missingdf)==1 and missingdf['Subject'].iloc[0]=='MDEM127':
	print('MDEM127 Did not complete both blocks of experiment\n'+
	'And therefore has incomplete PC data')
	combodf = combodf.drop(combodf[combodf['Subject']=='MDEM127'].index)

# High performing PC subjects:
excludedf = PCdf[(PCdf['Accuracy']>0.95) & (PCdf['Accuracy']<0.30)]
exclude_subjs = excludedf['Subject']

titlelist = ['Target (Correct)', 'Lure (Similar)', 'Foil']
pslabeldict = {'Target': 'Target (Correct) Selection Rate', 'Lure': 'Lure (Similar) Selection Rate', 'Foil': 'Foil Selection Rate'}
pclabeldict = {'Accuracy':'Relational Binding','Dependency':'Holistic Recollection'}
delaydict = {False:'Immediate',True:'Delayed'}
#combodf.rename(columns=labeldict, inplace=True)

statlist = []
for exclude in [True]:#,False]:
	for pc in ['Accuracy']:#,'Dependency']:
		f1, ax1 = plt.subplots(1,3,figsize=(15, 5))
		f2, ax2 = plt.subplots(1,3,figsize=(18, 6))
		for delay in [False,True]:
			for i,ps in enumerate(['Target','Lure','Foil']):
				tmp = combodf[combodf['Delay']==delay]
				#ps = labeldict[ps_]
				if exclude == True: tmp = tmp[~tmp.Subject.isin(exclude_subjs)]
				print('Delay = '+str(delay))
				print('Removing good subjs = '+str(exclude))
				print(ps+' x '+pc)
				print(len(tmp))
				res = stats.pearsonr(tmp[ps],tmp[pc])
				result = sm2.ols(formula=pc+'~Age +'+ps,data=tmp).fit()
				print(res)
				print(result.summary()) 
				tmpplt = tmp.rename(columns={ps: pslabeldict[ps]})
				tmpplt.rename(columns={pc: pclabeldict[pc]}, inplace=True)
				#fig, ax = plt.subplots(1, figsize=(5, 5))
				sns.regplot(data=tmpplt,x=pslabeldict[ps], y=pclabeldict[pc],ax=ax1[i],label=delaydict[delay])
				ax1[i].set_title(titlelist[i])
				statlist.append({'Controlling for Age':False,
								'Exclude':exclude,
								'PC':pc,
								'Delay':delay,
								'PS':ps,
								'r':res[0],
								'p':res[1]})
				#ax1[i].text(0.05, 0.8, f'r = {res[0]:.2f}\np = {res[1]:.3f}', transform=ax1[i].transAxes)
				
				# Partial regression plots controlling for age:
				PCreg = LinearRegression().fit(tmp[['Age']], tmp[pc])
				PSreg = LinearRegression().fit(tmp[['Age']], tmp[ps])
				PCresiduals = tmp[pc] - PCreg.predict(tmp[['Age']])
				PSresiduals = tmp[ps] - PSreg.predict(tmp[['Age']])
				residualsdf = pd.DataFrame({pslabeldict[ps]+' | Age': PSresiduals, 
											pclabeldict[pc]+' | Age': PCresiduals})
				# Partial regression plot
				#f, ax = plt.subplots(figsize=(6, 6))
				sns.regplot(data=residualsdf,x=pslabeldict[ps]+' | Age', y=pclabeldict[pc]+' | Age' ,ax=ax2[i],label=delaydict[delay])
				ax1[i].set_title(titlelist[i])
				# r-value from t-value:
				t_value = result.tvalues[ps]
				df = result.df_resid
				r_value = np.sqrt(np.square(t_value) / (np.square(t_value) + df))
				#ax2[i].text(0.05, 0.8, f'r = {r_value:.2f}\np = {result.pvalues[ps]:.3f}', transform=ax2[i].transAxes)
				statlist.append({'Controlling for Age':True,
								'Exclude':exclude,
								'PC':pc,
								'Delay':delay,
								'PS':ps,
								'r':r_value,
								'p':result.pvalues[ps]})
				if i>0:
					ax1[i].set(yticklabels=[])
					ax2[i].set(yticklabels=[])
					ax1[i].set(ylabel=None)
					ax2[i].set(ylabel=None)
		handles1, labels1 = ax1[i].get_legend_handles_labels()
		l1 = f1.legend(handles1, labels1, bbox_to_anchor=(1.16, .87), borderaxespad=0.)
		handles2, labels2 = ax2[i].get_legend_handles_labels()
		l2 = f2.legend(handles2, labels2, bbox_to_anchor=(1.13, .94), borderaxespad=0.)
		f1.tight_layout()
		f2.tight_layout()
		if pc == 'Accuracy':
			f1.savefig('Figures/Figure3.tif', dpi=300,bbox_inches="tight",format='tif')
			f1.show()
statdf=pd.DataFrame(statlist)
display(statdf)
				
				


		